//! Tools come from MCP servers (streamable HTTP, via `rmcp`); Ringmaster is
//! the MCP client and runs every tool call itself, so the servers stay on the
//! private network.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::header::{HeaderName, HeaderValue};
use rmcp::model::{CallToolRequestParams, ClientCapabilities, ClientConfig, Implementation};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::Mutex;

use crate::config::McpServerConfig;

/// A tool as offered by a server.
#[derive(Debug, Clone)]
pub struct DiscoveredTool {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Map<String, Value>,
}

impl DiscoveredTool {
    pub fn qualified(&self) -> String {
        format!("{}/{}", self.server, self.name)
    }
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool server {0} unreachable: {1}")]
    Unreachable(String, String),
    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait ToolBackend: Send + Sync {
    /// List the tools of the given servers. Fails if any of them is down.
    async fn list_tools(&self, servers: &[String]) -> Result<Vec<DiscoveredTool>, ToolError>;
    async fn call_tool(
        &self,
        server: &str,
        tool: &str,
        args: Map<String, Value>,
    ) -> Result<ToolOutput, ToolError>;
    fn server_names(&self) -> Vec<String>;
    /// Periodic clean-up (the worker calls it on every housekeeping pass):
    /// close sessions that are idle or dead.
    async fn housekeeping(&self) {}
}

/// A tool definition frozen into a job when its conversation starts, so the
/// `tools` array stays byte-identical for every turn (prompt caching and
/// preserved thinking both depend on it).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDef {
    /// Name Claude sees: `<server>__<tool>`, sanitised.
    pub claude_name: String,
    pub server: String,
    pub tool: String,
    pub description: String,
    pub input_schema: Value,
}

impl ToolDef {
    pub fn qualified(&self) -> String {
        format!("{}/{}", self.server, self.tool)
    }

    pub fn to_api(&self) -> Value {
        serde_json::json!({
            "name": self.claude_name,
            "description": self.description,
            "input_schema": self.input_schema,
        })
    }
}

/// Build sorted, uniquely named Claude tool definitions.
pub fn to_defs(tools: Vec<DiscoveredTool>) -> Vec<ToolDef> {
    let mut defs: Vec<ToolDef> = Vec::with_capacity(tools.len());
    let mut used = std::collections::HashSet::new();
    let mut tools = tools;
    tools.sort_by(|a, b| (&a.server, &a.name).cmp(&(&b.server, &b.name)));
    for t in tools {
        let mut base: String = format!("{}__{}", t.server, t.name)
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        base.truncate(64);
        let mut name = base.clone();
        let mut n = 2;
        while !used.insert(name.clone()) {
            let suffix = format!("_{n}");
            let mut b = base.clone();
            b.truncate(64 - suffix.len());
            name = format!("{b}{suffix}");
            n += 1;
        }
        let mut schema = t.input_schema;
        schema
            .entry("type")
            .or_insert_with(|| Value::String("object".into()));
        let description = if t.description.trim().is_empty() {
            format!("{} tool from the {} server", t.name, t.server)
        } else {
            t.description
        };
        defs.push(ToolDef {
            claude_name: name,
            server: t.server,
            tool: t.name,
            description,
            input_schema: Value::Object(schema),
        });
    }
    defs.sort_by(|a, b| a.claude_name.cmp(&b.claude_name));
    defs
}

/// Reject input schemas the Claude API would refuse, so one badly behaved MCP
/// server cannot break every request. This is a shape check, not a full JSON
/// Schema validation; anything subtler is caught by the agent's fallback of
/// retrying without tools when the API rejects the tool list.
pub fn check_schema(schema: &Map<String, Value>) -> Result<(), String> {
    match schema.get("type") {
        None => {}
        Some(Value::String(t)) if t == "object" => {}
        Some(other) => return Err(format!("input schema type is {other}, not \"object\"")),
    }
    match schema.get("properties") {
        None => {}
        Some(Value::Object(props)) => {
            if let Some((k, _)) = props
                .iter()
                .find(|(_, v)| !matches!(v, Value::Object(_) | Value::Bool(_)))
            {
                return Err(format!("property {k:?} is not a schema"));
            }
        }
        Some(_) => return Err("\"properties\" is not an object".into()),
    }
    match schema.get("required") {
        None => {}
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => {}
        Some(_) => return Err("\"required\" is not a list of names".into()),
    }
    Ok(())
}

type Client = RunningService<RoleClient, ClientConfig>;

/// MCP sessions unused for this long are closed by housekeeping; the next
/// call opens a new one. Commands come a few times a day, so this keeps no
/// session (and its rmcp worker task and SSE stream) open for weeks against
/// a server that may have restarted.
pub const MCP_SESSION_IDLE: Duration = Duration::from_secs(600);

/// Longest a session may take to close (rmcp also sends the server a
/// session DELETE); after that it is dropped, which cancels its tasks.
const CLOSE_LIMIT: Duration = Duration::from_secs(5);

struct Session {
    client: Arc<Client>,
    last_used: std::time::Instant,
}

/// MCP servers over streamable HTTP. Sessions are opened lazily, kept while
/// in use, re-opened once if a call fails on a dead session, and closed when
/// idle for [`MCP_SESSION_IDLE`] or dead. Each session owns rmcp tasks (a
/// service loop and a transport worker) that end when it is closed or the
/// last `Arc` to it is dropped (rmcp's drop guard cancels them).
pub struct McpBackend {
    servers: Vec<McpServerConfig>,
    /// Per-server request headers (credentials included), already validated.
    headers: HashMap<String, HashMap<HeaderName, HeaderValue>>,
    timeout: Duration,
    clients: Mutex<HashMap<String, Session>>,
}

/// Close a session within [`CLOSE_LIMIT`]. If a job still holds it, it is
/// closed when that job lets go of it (the drop cancels it).
async fn close_session(client: Arc<Client>) {
    if let Ok(client) = Arc::try_unwrap(client) {
        // Best effort; the session may already be gone.
        let _ = tokio::time::timeout(CLOSE_LIMIT, client.cancel()).await;
    }
}

impl McpBackend {
    pub fn new(
        servers: Vec<McpServerConfig>,
        headers: crate::config::McpHeaders,
        timeout: Duration,
    ) -> Self {
        let headers = headers
            .into_iter()
            .map(|(server, list)| {
                let map = list
                    .into_iter()
                    .filter_map(|(n, v)| {
                        let name = HeaderName::from_bytes(n.as_bytes()).ok()?;
                        let mut value = HeaderValue::from_str(&v).ok()?;
                        // Keep credentials out of debug output.
                        value.set_sensitive(true);
                        Some((name, value))
                    })
                    .collect();
                (server, map)
            })
            .collect();
        Self {
            servers,
            headers,
            timeout,
            clients: Mutex::new(HashMap::new()),
        }
    }

    async fn connect(&self, name: &str) -> Result<Arc<Client>> {
        let cfg = self
            .servers
            .iter()
            .find(|s| s.name == name)
            .with_context(|| format!("unknown MCP server {name}"))?;
        let mut tcfg = StreamableHttpClientTransportConfig::with_uri(cfg.url.clone())
            .reinit_on_expired_session(true);
        if let Some(h) = self.headers.get(name) {
            tcfg = tcfg.custom_headers(h.clone());
        }
        let transport = StreamableHttpClientTransport::from_config(tcfg);
        let info = ClientConfig::new(
            ClientCapabilities::default(),
            Implementation::new("ringmaster", env!("CARGO_PKG_VERSION")),
        );
        let client = tokio::time::timeout(self.timeout, info.serve(transport))
            .await
            .map_err(|_| anyhow!("timed out connecting"))?
            .map_err(|e| anyhow!("{e}"))?;
        Ok(Arc::new(client))
    }

    async fn client(&self, name: &str, fresh: bool) -> Result<Arc<Client>> {
        let mut clients = self.clients.lock().await;
        if !fresh && let Some(s) = clients.get_mut(name) {
            s.last_used = std::time::Instant::now();
            return Ok(s.client.clone());
        }
        let old = clients.remove(name);
        // Bounded by `timeout`; holding the lock avoids two sessions at once.
        let c = self.connect(name).await;
        if let Ok(c) = &c {
            clients.insert(
                name.to_string(),
                Session {
                    client: c.clone(),
                    last_used: std::time::Instant::now(),
                },
            );
        }
        self.set_gauge(clients.len());
        drop(clients);
        if let Some(old) = old {
            close_session(old.client).await;
        }
        c
    }

    fn set_gauge(&self, n: usize) {
        metrics::gauge!("ringmaster_mcp_sessions").set(n as f64);
    }

    /// Open sessions.
    pub async fn sessions(&self) -> usize {
        self.clients.lock().await.len()
    }

    /// Close sessions idle for longer than `max_idle`, and dead ones.
    /// Returns how many were closed.
    pub async fn close_idle(&self, max_idle: Duration) -> usize {
        let stale: Vec<Session> = {
            let mut clients = self.clients.lock().await;
            let names: Vec<String> = clients
                .iter()
                .filter(|(_, s)| s.last_used.elapsed() >= max_idle || s.client.is_closed())
                .map(|(n, _)| n.clone())
                .collect();
            let stale = names.iter().filter_map(|n| clients.remove(n)).collect();
            self.set_gauge(clients.len());
            stale
        };
        let n = stale.len();
        futures_util::future::join_all(stale.into_iter().map(|s| close_session(s.client))).await;
        if n > 0 {
            tracing::debug!(sessions = n, "closed idle MCP sessions");
        }
        n
    }

    /// `text` with this server's credential header values redacted (rmcp
    /// errors can quote what the server sent back).
    fn redact(&self, server: &str, text: &str) -> String {
        match self.headers.get(server) {
            Some(h) => {
                crate::config::redact_credentials(text, h.values().filter_map(|v| v.to_str().ok()))
            }
            None => text.to_string(),
        }
    }

    async fn list_one(&self, name: &str) -> Result<Vec<DiscoveredTool>> {
        let mut last = None;
        for fresh in [false, true] {
            let client = match self.client(name, fresh).await {
                Ok(c) => c,
                Err(e) => {
                    last = Some(e);
                    continue;
                }
            };
            match tokio::time::timeout(self.timeout, client.list_all_tools()).await {
                Ok(Ok(tools)) => {
                    return Ok(tools
                        .into_iter()
                        .map(|t| DiscoveredTool {
                            server: name.to_string(),
                            name: t.name.to_string(),
                            description: t.description.map(|d| d.to_string()).unwrap_or_default(),
                            input_schema: (*t.input_schema).clone(),
                        })
                        .collect());
                }
                Ok(Err(e)) => last = Some(anyhow!("{e}")),
                Err(_) => last = Some(anyhow!("timed out listing tools")),
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("unknown error")))
    }
}

#[async_trait]
impl ToolBackend for McpBackend {
    async fn list_tools(&self, servers: &[String]) -> Result<Vec<DiscoveredTool>, ToolError> {
        let mut all = Vec::new();
        for name in servers {
            let tools = self.list_one(name).await.map_err(|e| {
                ToolError::Unreachable(name.clone(), self.redact(name, &e.to_string()))
            })?;
            all.extend(tools);
        }
        Ok(all)
    }

    async fn call_tool(
        &self,
        server: &str,
        tool: &str,
        args: Map<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        // A call is only retried on a fresh session when the first attempt
        // could not reach the server at all; a call that reached it may have
        // had side effects.
        let mut last = None;
        for fresh in [false, true] {
            let client = match self.client(server, fresh).await {
                Ok(c) => c,
                Err(e) => {
                    last = Some(self.redact(server, &e.to_string()));
                    continue;
                }
            };
            let params = CallToolRequestParams::new(tool.to_string()).with_arguments(args.clone());
            match tokio::time::timeout(self.timeout, client.call_tool(params)).await {
                Ok(Ok(result)) => {
                    let is_error = result.is_error.unwrap_or(false);
                    let text = render_content(&result.content, result.structured_content.as_ref());
                    return Ok(ToolOutput { text, is_error });
                }
                Ok(Err(rmcp::ServiceError::McpError(e))) => {
                    return Ok(ToolOutput {
                        text: self
                            .redact(server, &format!("MCP error {}: {}", e.code.0, e.message)),
                        is_error: true,
                    });
                }
                Ok(Err(e @ rmcp::ServiceError::TransportClosed))
                | Ok(Err(e @ rmcp::ServiceError::TransportSend(_)))
                    if !fresh =>
                {
                    last = Some(self.redact(server, &e.to_string()));
                    continue;
                }
                Ok(Err(e)) => return Err(ToolError::Other(self.redact(server, &e.to_string()))),
                Err(_) => {
                    return Err(ToolError::Other(format!(
                        "timed out after {}s",
                        self.timeout.as_secs()
                    )));
                }
            }
        }
        Err(ToolError::Unreachable(
            server.to_string(),
            last.unwrap_or_default(),
        ))
    }

    fn server_names(&self) -> Vec<String> {
        self.servers.iter().map(|s| s.name.clone()).collect()
    }

    async fn housekeeping(&self) {
        self.close_idle(MCP_SESSION_IDLE).await;
    }
}

/// Flatten MCP content blocks into text for a `tool_result`.
fn render_content<T: Serialize>(content: &[T], structured: Option<&Value>) -> String {
    let mut parts = Vec::new();
    for block in content {
        let v = serde_json::to_value(block).unwrap_or(Value::Null);
        match v.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = v.get("text").and_then(Value::as_str) {
                    parts.push(t.to_string());
                }
            }
            Some("resource") => {
                if let Some(t) = v.pointer("/resource/text").and_then(Value::as_str) {
                    parts.push(t.to_string());
                } else {
                    parts.push(v.to_string());
                }
            }
            Some(other) => parts.push(format!("[{other} content omitted]")),
            None => {}
        }
    }
    if parts.is_empty()
        && let Some(s) = structured
    {
        parts.push(s.to_string());
    }
    if parts.is_empty() {
        "(no output)".into()
    } else {
        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mcp_errors_are_redacted_and_headers_sensitive() {
        let b = McpBackend::new(
            vec![],
            [(
                "home".to_string(),
                vec![
                    (
                        "Authorization".to_string(),
                        "Bearer mcp-tok-123456".to_string(),
                    ),
                    ("X-Api-Key".to_string(), "mcp-key-abcdef".to_string()),
                ],
            )]
            .into(),
            Duration::from_secs(1),
        );
        assert!(b.headers["home"].values().all(HeaderValue::is_sensitive));
        assert!(!format!("{:?}", b.headers).contains("mcp-tok-123456"));
        assert_eq!(
            b.redact("home", "401: got Bearer mcp-tok-123456 and mcp-key-abcdef"),
            "401: got [redacted] and [redacted]"
        );
        assert_eq!(b.redact("other", "mcp-key-abcdef"), "mcp-key-abcdef");
    }

    fn tool(server: &str, name: &str) -> DiscoveredTool {
        DiscoveredTool {
            server: server.into(),
            name: name.into(),
            description: String::new(),
            input_schema: Map::new(),
        }
    }

    #[test]
    fn schema_shape_checks() {
        let ok = |v: Value| check_schema(v.as_object().unwrap());
        assert!(ok(json!({})).is_ok());
        assert!(ok(json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]})).is_ok());
        assert!(ok(json!({"type": "array"})).is_err());
        assert!(ok(json!({"properties": []})).is_err());
        assert!(ok(json!({"properties": {"a": 3}})).is_err());
        assert!(ok(json!({"required": "a"})).is_err());
    }

    #[test]
    fn defs_are_sorted_sanitised_and_unique() {
        let defs = to_defs(vec![
            tool("homey", "set.light"),
            tool("homey", "set_light"),
            tool("deck", "z"),
            tool("abc", &"x".repeat(100)),
        ]);
        let names: Vec<_> = defs.iter().map(|d| d.claude_name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert!(names.contains(&"homey__set_light"));
        assert!(names.contains(&"homey__set_light_2"));
        assert!(names.iter().all(|n| n.len() <= 64));
        let d = defs.iter().find(|d| d.server == "deck").unwrap();
        assert_eq!(d.input_schema["type"], "object");
        assert!(!d.description.is_empty());
    }

    #[test]
    fn renders_text_and_structured() {
        let content = vec![
            serde_json::json!({"type": "text", "text": "a"}),
            serde_json::json!({"type": "image", "data": "..", "mimeType": "image/png"}),
        ];
        assert_eq!(render_content(&content, None), "a\n[image content omitted]");
        let empty: Vec<Value> = vec![];
        assert_eq!(
            render_content(&empty, Some(&serde_json::json!({"ok": true}))),
            "{\"ok\":true}"
        );
    }
}
