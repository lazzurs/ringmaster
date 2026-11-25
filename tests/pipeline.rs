//! End-to-end tests: signed webhook -> queue -> whisper (mock) -> LLM
//! (wiremock: Claude, OpenAI, Mistral, OpenRouter, Ollama) -> MCP tools (a
//! real rmcp server on localhost) -> notification, including the
//! confirm/deny flow.
//!
//! Run with proxy variables unset if your shell has them.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerConfig};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData as McpError, ServerHandler, schemars, tool, tool_handler, tool_router};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path, path_regex};
use wiremock::{Mock, MockServer, Request as MockRequest, Respond, ResponseTemplate};

use ringmaster::agent::Agent;
use ringmaster::config::{Config, McpServerConfig};
use ringmaster::db::{Db, JobStatus};
use ringmaster::http::{AppState, router};
use ringmaster::llm::{Llms, Provider};
use ringmaster::notify::{Kind, Notification, Notifier};
use ringmaster::signature;
use ringmaster::stt::Transcriber;
use ringmaster::tools::{DiscoveredTool, McpBackend, ToolBackend, ToolError, ToolOutput};
use ringmaster::worker::Worker;

const SECRET: &str = "0123456789abcdef0123456789abcdef-test";
const TONE: &[u8] = include_bytes!("fixtures/tone-1s.m4a");

// ------------------------------------------------------------ mock MCP server

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct LightArgs {
    /// Light name
    name: String,
    on: bool,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct DoorArgs {
    door: String,
}

#[derive(Clone)]
struct Home {
    calls: Arc<Mutex<Vec<String>>>,
}

#[tool_router]
impl Home {
    #[tool(description = "Turn a light on or off")]
    async fn set_light(
        &self,
        Parameters(a): Parameters<LightArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("set_light {} {}", a.name, a.on));
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "{} is now {}",
            a.name,
            if a.on { "on" } else { "off" }
        ))]))
    }

    #[tool(description = "Unlock a door")]
    async fn unlock_door(
        &self,
        Parameters(a): Parameters<DoorArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("unlock_door {}", a.door));
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "unlocked",
        )]))
    }

    #[tool(description = "Delete every device")]
    async fn delete_everything(&self) -> Result<CallToolResult, McpError> {
        self.calls.lock().unwrap().push("delete_everything".into());
        Ok(CallToolResult::success(vec![ContentBlock::text("gone")]))
    }
}

#[tool_handler]
impl ServerHandler for Home {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}

async fn start_mcp() -> (String, Arc<Mutex<Vec<String>>>, CancellationToken) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let home = Home {
        calls: calls.clone(),
    };
    let ct = CancellationToken::new();
    let service = StreamableHttpService::new(
        move || Ok(home.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(ct.child_token()),
    );
    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stop = ct.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { stop.cancelled().await })
            .await
            .unwrap();
    });
    (format!("http://{addr}/mcp"), calls, ct)
}

// ------------------------------------------------------------- mock Claude

/// Scripted model API: returns `turns[n]` for the n-th request and records
/// every request body, plus its path and headers.
#[derive(Clone)]
struct ScriptedClaude {
    turns: Arc<Vec<Value>>,
    seen: Arc<Mutex<Vec<Value>>>,
    meta: Arc<Mutex<Vec<Value>>>,
}

impl Respond for ScriptedClaude {
    fn respond(&self, req: &MockRequest) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let headers: serde_json::Map<String, Value> = req
            .headers
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), json!(v.to_str().unwrap_or(""))))
            .collect();
        self.meta
            .lock()
            .unwrap()
            .push(json!({"path": req.url.path(), "headers": headers}));
        let mut seen = self.seen.lock().unwrap();
        let n = seen.len();
        seen.push(body);
        match self.turns.get(n) {
            // `{"__status": 400, "__body": {...}}` scripts an API error.
            Some(t) if t.get("__status").is_some() => {
                ResponseTemplate::new(t["__status"].as_u64().unwrap() as u16)
                    .set_body_json(t["__body"].clone())
            }
            Some(t) => ResponseTemplate::new(200).set_body_json(t.clone()),
            None => ResponseTemplate::new(500).set_body_string("script exhausted"),
        }
    }
}

fn tool_use_turn(id: &str, name: &str, input: Value) -> Value {
    json!({
        "id": format!("msg_{id}"),
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "", "signature": format!("sig-{id}")},
            {"type": "tool_use", "id": id, "name": name, "input": input}
        ],
        "usage": {"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 800}
    })
}

fn final_turn(text: &str) -> Value {
    json!({
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 850, "cache_creation_input_tokens": 60}
    })
}

// ------------------------------------------------------------- notifier

#[derive(Default)]
struct Recorder(Mutex<Vec<Notification>>);

#[async_trait]
impl Notifier for Recorder {
    async fn send(&self, n: &Notification) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(n.clone());
        Ok(())
    }
}

impl Recorder {
    fn take(&self) -> Vec<Notification> {
        std::mem::take(&mut self.0.lock().unwrap())
    }
}

// ------------------------------------------------------------- harness

struct Harness {
    app: axum::Router,
    worker: Arc<Worker>,
    db: Db,
    notes: Arc<Recorder>,
    claude_seen: Arc<Mutex<Vec<Value>>>,
    /// Path and headers of each model request.
    llm_meta: Arc<Mutex<Vec<Value>>>,
    db_path: std::path::PathBuf,
    mcp_calls: Arc<Mutex<Vec<String>>>,
    _mcp_stop: CancellationToken,
    _servers: Vec<MockServer>,
    _dir: tempfile::TempDir,
}

const CONFIG: &str = r#"
[server]
public_base_url = "https://ring.test"

[stt]
vocabulary = ["kitchen"]

[claude]
model = "claude-opus-5-5"
effort = "low"
max_retries = 0

[agent]
timezone = "Europe/Dublin"
retry_base_secs = 1
tool_discovery_timeout_secs = 2

[[tool_policies]]
match = "home/*delete*"
policy = "deny"

[[tool_policies]]
match = "home/unlock_*"
policy = "confirm"

[[tool_policies]]
match = "home/*"
policy = "allow"

[profiles.command]
triggers = ["double-click-hold"]
tools = ["*/*"]
system_prompt = "Command mode."

[profiles.capture]
triggers = ["single-click-hold"]
tools = ["home/set_light"]
system_prompt = "Capture mode."
"#;

async fn harness(turns: Vec<Value>, whisper_text: Option<&str>) -> Harness {
    harness_with(turns, whisper_text, |home| vec![home]).await
}

/// `servers` gets the live test MCP server ("home") and returns the list the
/// service is configured with.
async fn harness_with(
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
) -> Harness {
    harness_config(CONFIG, turns, whisper_text, servers).await
}

/// The test config with its `[claude]` section replaced by `llm`.
fn config_with(llm: &str) -> String {
    let legacy = "[claude]\nmodel = \"claude-opus-5-5\"\neffort = \"low\"\nmax_retries = 0\n";
    assert!(CONFIG.contains(legacy));
    CONFIG.replace(legacy, llm)
}

async fn harness_config(
    config_text: &str,
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
) -> Harness {
    harness_full(config_text, turns, whisper_text, servers, |t| t).await
}

/// `harness_config`, with the MCP backend wrapped by `wrap_tools`.
async fn harness_full(
    config_text: &str,
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
    wrap_tools: impl FnOnce(Arc<dyn ToolBackend>) -> Arc<dyn ToolBackend>,
) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let (mcp_url, mcp_calls, mcp_stop) = start_mcp().await;

    let whisper = MockServer::start().await;
    match whisper_text {
        Some(t) => {
            Mock::given(method("POST"))
                .and(path("/inference"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": t})))
                .mount(&whisper)
                .await
        }
        None => {
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(503))
                .mount(&whisper)
                .await
        }
    }

    let claude_server = MockServer::start().await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let meta = Arc::new(Mutex::new(Vec::new()));
    // Every provider's endpoint, on one scripted server.
    Mock::given(method("POST"))
        .and(path_regex(r"^/(v1/messages|v1/chat/completions|api/chat)$"))
        .respond_with(ScriptedClaude {
            turns: Arc::new(turns),
            seen: seen.clone(),
            meta: meta.clone(),
        })
        .mount(&claude_server)
        .await;

    let mut config = Config::from_toml(config_text).unwrap();
    config.server.data_dir = dir.path().to_path_buf();
    config.stt.primary_url = Some(whisper.uri());
    config.claude.api_base = claude_server.uri();
    for p in [
        &mut config.llm.openai,
        &mut config.llm.mistral,
        &mut config.llm.openrouter,
        &mut config.llm.ollama,
    ] {
        p.base_url = Some(claude_server.uri());
    }
    config.mcp_servers = servers(McpServerConfig {
        name: "home".into(),
        url: mcp_url,
        ..Default::default()
    });
    let config = Arc::new(config);

    let db = Db::open(&dir.path().join("test.db")).unwrap();
    let notes = Arc::new(Recorder::default());
    let notifier: Arc<dyn Notifier> = notes.clone();
    let keys = [
        (Provider::Anthropic, "sk-test"),
        (Provider::OpenAi, "sk-openai-test"),
        (Provider::Mistral, "mistral-test"),
        (Provider::OpenRouter, "sk-or-test"),
    ]
    .into_iter()
    .map(|(p, k)| (p, k.to_string()))
    .collect();
    let llms = Llms::new(&config, &keys).unwrap();
    let tools = wrap_tools(Arc::new(McpBackend::new(
        config.mcp_servers.clone(),
        Default::default(),
        Duration::from_secs(10),
    )));
    let agent =
        Arc::new(Agent::new(config.clone(), db.clone(), llms, tools, notifier.clone()).unwrap());
    let wake = Arc::new(Notify::new());
    let worker = Arc::new(Worker {
        config: config.clone(),
        db: db.clone(),
        transcriber: Transcriber::new(config.stt.clone()).unwrap(),
        agent,
        notifier: notifier.clone(),
        wake: wake.clone(),
        data_dir: dir.path().to_path_buf(),
    });
    let state = Arc::new(AppState {
        config: config.clone(),
        db: db.clone(),
        notifier,
        wake,
        data_dir: dir.path().to_path_buf(),
        webhook_secret: Some(SECRET.into()),
        metrics: None,
    });
    Harness {
        app: router(state),
        worker,
        db,
        notes,
        claude_seen: seen,
        llm_meta: meta,
        db_path: dir.path().join("test.db"),
        mcp_calls,
        _mcp_stop: mcp_stop,
        _servers: vec![whisper, claude_server],
        _dir: dir,
    }
}

fn multipart(audio: Option<&[u8]>, transcription: Option<&str>, test: bool) -> (String, Vec<u8>) {
    let boundary = "ringmaster-test-boundary";
    let mut body = Vec::new();
    if let Some(a) = audio {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"rec.m4a\"\r\nContent-Type: audio/mp4\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(a);
        body.extend_from_slice(b"\r\n");
    }
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    };
    if let Some(t) = transcription {
        field("transcription", t);
    }
    if test {
        field("test", "true");
    }
    field(
        "recordedAt",
        &(jiff::Timestamp::now().as_millisecond()).to_string(),
    );
    field("client", "ring");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn signed_request(
    delivery: &str,
    trigger: &str,
    test: bool,
    ct: &str,
    body: Vec<u8>,
    ts: i64,
) -> Request<Body> {
    let ts = ts.to_string();
    let sig = signature::sign(SECRET.as_bytes(), &ts, delivery, trigger, test, &body);
    let mut b = Request::post("/webhook")
        .header("content-type", ct)
        .header("x-index-webhook-version", "1")
        .header("x-index-trigger", trigger)
        .header("x-index-delivery", delivery)
        .header("x-index-timestamp", ts)
        .header("x-index-signature", sig)
        .header("x-audio-size", body.len().to_string());
    if test {
        b = b.header("x-index-test", "true");
    }
    b.body(Body::from(body)).unwrap()
}

async fn send(h: &Harness, req: Request<Body>) -> (StatusCode, String) {
    let resp = h.app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn now() -> i64 {
    jiff::Timestamp::now().as_second()
}

async fn tick(h: &Harness) {
    h.worker.tick(&CancellationToken::new()).await;
}

// ------------------------------------------------------------- tests

#[tokio::test]
async fn command_runs_tool_and_reports_result() {
    let h = harness(
        vec![
            tool_use_turn(
                "toolu_1",
                "home__set_light",
                json!({"name": "kitchen", "on": true}),
            ),
            final_turn("Kitchen lights on."),
        ],
        Some(" Turn on the kitchen lights."),
    )
    .await;

    let (ct, body) = multipart(Some(TONE), Some("turn on the kitten lights"), false);
    let (status, text) = send(
        &h,
        signed_request(
            "rec-1",
            "double-click-hold",
            false,
            &ct,
            body.clone(),
            now(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
    assert!(text.contains("queued"));

    // The phone retries: acknowledged, not reprocessed.
    let (status, text) = send(
        &h,
        signed_request("rec-1", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("duplicate"));

    tick(&h).await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert_eq!(notes[0].body, "Kitchen lights on.");

    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let first = &seen[0];
    // Request shape for Opus 5.5.
    assert_eq!(first["model"], "claude-opus-5-5");
    assert_eq!(first["output_config"]["effort"], "low");
    assert_eq!(first["fallbacks"], "default");
    assert_eq!(first["cache_control"]["type"], "ephemeral");
    assert_eq!(first["system"][0]["cache_control"]["type"], "ephemeral");
    for forbidden in ["thinking", "tool_choice", "temperature", "top_p", "top_k"] {
        assert!(
            first.get(forbidden).is_none(),
            "{forbidden} must not be sent"
        );
    }
    // Tools: sorted, denied ones hidden.
    let names: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["home__set_light", "home__unlock_door"]);
    // Volatile context is in the first user message, not the system prompt.
    let user_text = first["messages"][0]["content"][0]["text"].as_str().unwrap();
    assert!(
        user_text.contains("Turn on the kitchen lights."),
        "{user_text}"
    );
    assert!(user_text.contains("Transcript source: primary"));
    assert!(user_text.contains("Current time:"));
    assert!(
        !first["system"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Current time")
    );

    let second = &seen[1];
    // Prefix unchanged: same system and tools, history appended.
    assert_eq!(second["system"], first["system"]);
    assert_eq!(second["tools"], first["tools"]);
    assert_eq!(second["messages"][0], first["messages"][0]);
    // Assistant turn echoed back unchanged, thinking block included.
    assert_eq!(second["messages"][1]["role"], "assistant");
    assert_eq!(
        second["messages"][1]["content"][0]["signature"],
        "sig-toolu_1"
    );
    // All tool results in one user message.
    let results = &second["messages"][2];
    assert_eq!(results["role"], "user");
    assert_eq!(results["content"][0]["type"], "tool_result");
    assert_eq!(results["content"][0]["tool_use_id"], "toolu_1");
    assert_eq!(results["content"][0]["content"], "kitchen is now on");
    assert!(results["content"][0].get("is_error").is_none());

    // Audit trail.
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Done);
    assert_eq!(job.transcript_source.as_deref(), Some("primary"));
    let kinds: Vec<String> =
        h.db.audit_entries(1)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
    for k in [
        "delivery",
        "transcript",
        "agent_start",
        "claude_response",
        "tool_call",
        "job_finished",
    ] {
        assert!(kinds.iter().any(|x| x == k), "missing audit {k}: {kinds:?}");
    }
}

#[tokio::test]
async fn confirm_approve_resumes_and_runs_tool() {
    let h = harness(
        vec![
            tool_use_turn("toolu_9", "home__unlock_door", json!({"door": "front"})),
            final_turn("Front door unlocked."),
        ],
        Some("unlock the front door"),
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-2", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);

    tick(&h).await;
    assert!(
        h.mcp_calls.lock().unwrap().is_empty(),
        "must not run before approval"
    );
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::AwaitingConfirmation);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    let Kind::Confirm {
        approve_url: Some(approve),
        view_url: Some(view),
        ..
    } = &notes[0].kind
    else {
        panic!("expected confirm notification: {:?}", notes[0]);
    };
    assert!(notes[0].body.contains("home/unlock_door"));
    let approve_path = approve
        .strip_prefix("https://ring.test")
        .unwrap()
        .to_string();
    let view_path = view.strip_prefix("https://ring.test").unwrap().to_string();

    // GET shows the page and changes nothing.
    let (status, page) = send(&h, Request::get(&view_path).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(page.contains("home/unlock_door"));
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );

    let (status, _) = send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // One-time token.
    let (status, _) = send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::GONE);

    tick(&h).await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door front".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.last().unwrap().body, "Front door unlocked.");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1]["messages"][2]["content"][0]["content"], "unlocked");
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::Done
    );
}

#[tokio::test]
async fn confirm_deny_returns_error_result_without_running() {
    let h = harness(
        vec![
            tool_use_turn("toolu_5", "home__unlock_door", json!({"door": "back"})),
            final_turn("OK, I left the back door locked."),
        ],
        Some("unlock the back door"),
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-3", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;
    let notes = h.notes.take();
    let Kind::Confirm {
        deny_url: Some(deny),
        ..
    } = &notes[0].kind
    else {
        panic!()
    };
    let deny_path = deny.strip_prefix("https://ring.test").unwrap().to_string();
    let (status, _) = send(&h, Request::post(&deny_path).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);

    tick(&h).await;
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let seen = h.claude_seen.lock().unwrap().clone();
    let result = &seen[1]["messages"][2]["content"][0];
    assert_eq!(result["is_error"], true);
    assert!(result["content"].as_str().unwrap().contains("declined"));
    assert_eq!(
        h.notes.take().last().unwrap().body,
        "OK, I left the back door locked."
    );
}

#[tokio::test]
async fn profile_allow_list_limits_tools() {
    let h = harness(vec![final_turn("Added.")], Some("note: buy milk")).await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-4", "single-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    tick(&h).await;
    let seen = h.claude_seen.lock().unwrap().clone();
    let names: Vec<&str> = seen[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["home__set_light"]);
    assert!(
        seen[0]["system"][0]["text"]
            .as_str()
            .unwrap()
            .ends_with("Capture mode.")
    );
}

/// A server nothing is listening on.
fn dead_server(name: &str) -> McpServerConfig {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    McpServerConfig {
        name: name.into(),
        url: format!("http://127.0.0.1:{port}/mcp"),
        ..Default::default()
    }
}

#[tokio::test]
async fn wildcard_profile_skips_unavailable_server() {
    let h = harness_with(
        vec![final_turn("Done.")],
        Some("Turn on the kitchen lights."),
        |home| vec![home, dead_server("homey")],
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-dead-1", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    tick(&h).await;

    // The job ran with the live server's tools and was told what is missing.
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "Claude should have been called once");
    let names: Vec<&str> = seen[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.iter().all(|n| n.starts_with("home__")), "{names:?}");
    let first = seen[0]["messages"][0]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(first.contains("Unavailable right now"), "{first}");
    assert!(first.contains("homey"), "{first}");

    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(notes[0].body.starts_with("Done."), "{}", notes[0].body);
    assert!(
        notes[0].body.contains("Unavailable: homey"),
        "{}",
        notes[0].body
    );
    assert!(
        notes[0].body.contains("Turn on the kitchen lights."),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn only_server_down_still_answers_and_keeps_the_request() {
    // The capture profile names `home/...` and that server is down: the job
    // still runs (with no tools), and the reply carries what the user said.
    let h = harness_with(
        vec![final_turn(
            "I couldn't add that: the task list is unavailable.",
        )],
        Some("buy milk"),
        |_| vec![dead_server("home")],
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-dead-2", "single-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    tick(&h).await;

    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].get("tools").is_none(), "no tools to offer");
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Done);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(
        notes[0].body.contains("Unavailable: home"),
        "{}",
        notes[0].body
    );
    assert!(notes[0].body.contains("buy milk"), "{}", notes[0].body);
}

#[tokio::test]
async fn rejected_tool_definitions_fall_back_to_no_tools() {
    let h = harness(
        vec![
            json!({"__status": 400, "__body": {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "tools.0.input_schema: invalid"}
            }}),
            final_turn("Sorry, I can't do that right now."),
        ],
        Some("Turn on the kitchen lights."),
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-400", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;

    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "retried once");
    assert!(seen[0].get("tools").is_some());
    assert!(seen[1].get("tools").is_none(), "second try has no tools");
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(
        notes[0].body.contains("Unavailable: all tools"),
        "{}",
        notes[0].body
    );
    assert!(
        notes[0].body.contains("kitchen lights"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn hung_server_is_skipped_quickly() {
    // Accepts connections but never answers.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((sock, _)) = listener.accept().await {
            held.push(sock);
        }
    });
    let h = harness_with(vec![final_turn("Done.")], Some("lights on"), move |home| {
        vec![
            home,
            McpServerConfig {
                name: "slow".into(),
                url: format!("http://127.0.0.1:{port}/mcp"),
                ..Default::default()
            },
        ]
    })
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-hung", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    let started = std::time::Instant::now();
    tick(&h).await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "took {:?}",
        started.elapsed()
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(
        notes[0].body.contains("Unavailable: slow"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn mcp_requests_carry_configured_headers() {
    use ringmaster::tools::ToolBackend;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("authorization", "Basic cm9iOmFwcC1wYXNz"))
        .and(header("x-extra", "v"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1..)
        .mount(&server)
        .await;
    let backend = McpBackend::new(
        vec![McpServerConfig {
            name: "nc".into(),
            url: format!("{}/mcp", server.uri()),
            ..Default::default()
        }],
        [(
            "nc".to_string(),
            vec![
                (
                    "Authorization".to_string(),
                    "Basic cm9iOmFwcC1wYXNz".to_string(),
                ),
                ("X-Extra".to_string(), "v".to_string()),
            ],
        )]
        .into(),
        Duration::from_secs(5),
    );
    // The mock answers 503, so listing fails; what matters is what was sent.
    assert!(backend.list_tools(&["nc".to_string()]).await.is_err());
    server.verify().await;
}

#[tokio::test]
async fn whisper_down_falls_back_to_phone_transcription() {
    let h = harness(vec![final_turn("Done it.")], None).await;
    let (ct, body) = multipart(Some(TONE), Some("lights off"), false);
    send(
        &h,
        signed_request("rec-5", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.transcript_source.as_deref(), Some("phone"));
    assert_eq!(job.transcript.as_deref(), Some("lights off"));
    assert_eq!(job.status, JobStatus::Done);
}

#[tokio::test]
async fn whisper_down_without_phone_text_retries_later() {
    let h = harness(vec![final_turn("never")], None).await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-6", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Queued);
    assert!(h.db.next_due().await.unwrap().unwrap() > now() - 1);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].kind, Kind::Error);
    assert!(notes[0].body.contains("keep trying"), "{}", notes[0].body);
    assert!(h.claude_seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn test_event_is_logged_not_run() {
    let h = harness(vec![], Some("x")).await;
    let (ct, body) = multipart(None, Some("This is a test"), true);
    let (status, text) = send(
        &h,
        signed_request("test-1", "test-event", true, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(text.contains("test event"));
    tick(&h).await;
    assert!(h.db.get_job(1).await.unwrap().is_none());
    assert!(h.claude_seen.lock().unwrap().is_empty());
    // The notification is sent in the background.
    for _ in 0..50 {
        if !h.notes.0.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(h.notes.take()[0].kind, Kind::Info);
}

#[tokio::test]
async fn rejects_bad_signature_stale_timestamp_and_unknown_version() {
    let h = harness(vec![], Some("x")).await;
    let (ct, body) = multipart(None, Some("hi"), false);

    let mut req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    req.headers_mut()
        .insert("x-index-signature", "0".repeat(64).parse().unwrap());
    assert_eq!(send(&h, req).await.0, StatusCode::UNAUTHORIZED);

    let req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now() - 301,
    );
    assert_eq!(send(&h, req).await.0, StatusCode::UNAUTHORIZED);

    let mut req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    req.headers_mut()
        .insert("x-index-webhook-version", "2".parse().unwrap());
    assert_eq!(send(&h, req).await.0, StatusCode::BAD_REQUEST);

    let mut req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    req.headers_mut().remove("x-index-signature");
    req.headers_mut().remove("x-index-timestamp");
    assert_eq!(send(&h, req).await.0, StatusCode::UNAUTHORIZED);

    // Nothing was stored by the rejected attempts; a good one still works.
    let req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    assert_eq!(send(&h, req).await.0, StatusCode::ACCEPTED);

    // Unmapped gesture.
    let req = signed_request("rec-8", "triple-click", false, &ct, body, now());
    assert_eq!(send(&h, req).await.0, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn health_endpoint() {
    let h = harness(vec![], Some("x")).await;
    let (status, body) = send(&h, Request::get("/healthz").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "ok");
}

// ------------------------------------------------- other providers

/// An OpenAI-style reply with one tool call (arguments as a JSON string).
fn chat_tool_turn(id: &str, name: &str, arguments: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "refusal": null,
                "tool_calls": [{"id": id, "type": "function",
                                "function": {"name": name, "arguments": arguments}}]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920}
    })
}

fn chat_final(text: &str) -> Value {
    json!({
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "refusal": null},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 950, "completion_tokens": 8, "total_tokens": 958}
    })
}

/// Ollama's native `/api/chat` reply: arguments as an object, no id.
fn ollama_tool_turn(name: &str, arguments: Value) -> Value {
    json!({
        "model": "qwen3:8b",
        "created_at": "2026-11-25T10:00:00Z",
        "message": {"role": "assistant", "content": "", "thinking": "The user wants light.",
                    "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 1200,
        "eval_count": 30
    })
}

fn ollama_final(text: &str) -> Value {
    json!({
        "model": "qwen3:8b",
        "created_at": "2026-11-25T10:00:05Z",
        "message": {"role": "assistant", "content": text},
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 1300,
        "eval_count": 12
    })
}

async fn run_command(h: &Harness, delivery: &str, trigger: &str) {
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, text) = send(
        h,
        signed_request(delivery, trigger, false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
    tick(h).await;
}

fn meta(h: &Harness) -> Vec<Value> {
    h.llm_meta.lock().unwrap().clone()
}

async fn job_provider(h: &Harness) -> Value {
    let job = h.db.get_job(1).await.unwrap().unwrap();
    serde_json::to_value(job.agent_state.unwrap()).unwrap()["provider"].clone()
}

#[tokio::test]
async fn openai_tool_loop_request_shape() {
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"openai\"\nmodel = \"gpt-6.2-sol\"\neffort = \"low\"\nmax_tokens = 2000\nmax_retries = 0\n",
        ),
        vec![
            chat_tool_turn(
                "call_abc",
                "home__set_light",
                r#"{"name":"kitchen","on":true}"#,
            ),
            chat_final("Kitchen lights on."),
        ],
        Some("Turn on the kitchen lights."),
        |home| vec![home],
    )
    .await;
    run_command(&h, "oa-1", "double-click-hold").await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert_eq!(notes[0].body, "Kitchen lights on.");

    let m = meta(&h);
    assert_eq!(m.len(), 2);
    assert_eq!(m[0]["path"], "/v1/chat/completions");
    assert_eq!(m[0]["headers"]["authorization"], "Bearer sk-openai-test");
    assert!(m[0]["headers"].get("x-api-key").is_none());

    let seen = h.claude_seen.lock().unwrap().clone();
    let first = &seen[0];
    assert_eq!(first["model"], "gpt-6.2-sol");
    assert_eq!(first["reasoning_effort"], "low");
    assert_eq!(first["max_completion_tokens"], 2000);
    assert!(first.get("max_tokens").is_none());
    for anthropic_only in ["system", "output_config", "cache_control", "fallbacks"] {
        assert!(first.get(anthropic_only).is_none(), "{anthropic_only}");
    }
    assert_eq!(first["messages"][0]["role"], "developer");
    assert!(
        first["messages"][0]["content"]
            .as_str()
            .unwrap()
            .ends_with("Command mode.")
    );
    assert_eq!(first["messages"][1]["role"], "user");
    assert!(
        first["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("Turn on the kitchen lights.")
    );
    let tools: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            assert_eq!(t["type"], "function");
            assert_eq!(t["function"]["parameters"]["type"], "object");
            t["function"]["name"].as_str().unwrap()
        })
        .collect();
    assert_eq!(tools, ["home__set_light", "home__unlock_door"]);

    let second = &seen[1];
    assert_eq!(second["tools"], first["tools"]);
    assert_eq!(second["messages"][0], first["messages"][0]);
    assert_eq!(second["messages"][1], first["messages"][1]);
    // The assistant turn is echoed with its tool call; response-only fields
    // (refusal) are not.
    assert_eq!(
        second["messages"][2],
        json!({"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc", "type": "function",
            "function": {"name": "home__set_light", "arguments": r#"{"name":"kitchen","on":true}"#}}]})
    );
    assert_eq!(
        second["messages"][3],
        json!({"role": "tool", "tool_call_id": "call_abc", "content": "kitchen is now on"})
    );

    assert_eq!(job_provider(&h).await, "openai");
    let kinds: Vec<String> =
        h.db.audit_entries(1)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
    assert!(kinds.iter().any(|k| k == "llm_response"), "{kinds:?}");
}

#[tokio::test]
async fn mistral_confirm_approve_round_trip() {
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"mistral\"\nmodel = \"mistral-small-latest\"\nmax_retries = 0\n",
        ),
        vec![
            chat_tool_turn("D681PevKs", "home__unlock_door", r#"{"door":"front"}"#),
            chat_final("Front door unlocked."),
        ],
        Some("unlock the front door"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "mi-1", "double-click-hold").await;
    assert!(
        h.mcp_calls.lock().unwrap().is_empty(),
        "must not run before approval"
    );
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    let notes = h.notes.take();
    let Kind::Confirm {
        approve_url: Some(approve),
        ..
    } = &notes[0].kind
    else {
        panic!("expected confirm notification: {:?}", notes[0]);
    };
    assert!(notes[0].body.contains("home/unlock_door"));
    let approve_path = approve
        .strip_prefix("https://ring.test")
        .unwrap()
        .to_string();
    let (status, _) = send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    tick(&h).await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door front".to_string()]
    );
    assert_eq!(h.notes.take().last().unwrap().body, "Front door unlocked.");
    let m = meta(&h);
    assert_eq!(m[0]["path"], "/v1/chat/completions");
    assert_eq!(m[0]["headers"]["authorization"], "Bearer mistral-test");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["model"], "mistral-small-latest");
    assert_eq!(seen[0]["messages"][0]["role"], "system");
    assert!(
        seen[0].get("reasoning_effort").is_none(),
        "no effort configured"
    );
    assert!(seen[0].get("max_tokens").is_none(), "no limit configured");
    assert_eq!(seen[1]["messages"][2]["tool_calls"][0]["id"], "D681PevKs");
    assert_eq!(
        seen[1]["messages"][3],
        json!({"role": "tool", "tool_call_id": "D681PevKs", "name": "home__unlock_door", "content": "unlocked"})
    );
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::Done
    );
}

#[tokio::test]
async fn openrouter_sends_attribution_and_echoes_reasoning() {
    let mut tool_turn = chat_tool_turn(
        "toolu_or1",
        "home__set_light",
        r#"{"name":"hall","on":false}"#,
    );
    let details = json!([{"type": "reasoning.encrypted", "data": "opaque", "id": "r1",
                          "format": "anthropic-claude-v1", "index": 0}]);
    tool_turn["choices"][0]["message"]["reasoning"] = json!("Turn off the hall light.");
    tool_turn["choices"][0]["message"]["reasoning_details"] = details.clone();
    tool_turn["choices"][0]["native_finish_reason"] = json!("tool_use");
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"openrouter\"\nmodel = \"anthropic/claude-sonnet-5.5\"\neffort = \"medium\"\nmax_retries = 0\n\n[llm.openrouter]\napp_url = \"https://github.com/lazzurs/ringmaster\"\napp_title = \"Ringmaster\"\n",
        ),
        vec![tool_turn, chat_final("Hall light off.")],
        Some("hall light off"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "or-1", "double-click-hold").await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light hall false".to_string()]
    );
    assert_eq!(h.notes.take()[0].body, "Hall light off.");
    let m = meta(&h);
    assert_eq!(m[0]["headers"]["authorization"], "Bearer sk-or-test");
    assert_eq!(
        m[0]["headers"]["http-referer"],
        "https://github.com/lazzurs/ringmaster"
    );
    assert_eq!(m[0]["headers"]["x-openrouter-title"], "Ringmaster");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen[0]["reasoning"], json!({"effort": "medium"}));
    assert!(seen[0].get("reasoning_effort").is_none());
    let echoed = &seen[1]["messages"][2];
    assert_eq!(
        echoed["reasoning_details"], details,
        "passed back unchanged"
    );
    assert!(echoed.get("reasoning").is_none());
    assert_eq!(seen[1]["messages"][3]["tool_call_id"], "toolu_or1");
}

#[tokio::test]
async fn ollama_native_tool_loop_without_auth() {
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\neffort = \"none\"\nmax_retries = 0\n",
        ),
        vec![
            ollama_tool_turn("home__set_light", json!({"name": "kitchen", "on": true})),
            ollama_final("Kitchen light is on."),
        ],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "ol-1", "double-click-hold").await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    assert_eq!(h.notes.take()[0].body, "Kitchen light is on.");

    let m = meta(&h);
    assert_eq!(m[0]["path"], "/api/chat");
    assert!(
        m[0]["headers"].get("authorization").is_none(),
        "Ollama needs no key"
    );
    let seen = h.claude_seen.lock().unwrap().clone();
    let first = &seen[0];
    assert_eq!(first["model"], "qwen3:8b");
    assert_eq!(first["stream"], false);
    assert_eq!(first["truncate"], false);
    assert_eq!(first["options"]["num_ctx"], 8192);
    assert_eq!(first["think"], false);
    assert_eq!(first["messages"][0]["role"], "system");
    assert_eq!(first["tools"][0]["type"], "function");

    // Object arguments, a generated id and the thinking are echoed back.
    let echoed = &seen[1]["messages"][2];
    let id = echoed["tool_calls"][0]["id"].as_str().unwrap();
    assert_eq!(id.len(), 9);
    assert_eq!(
        echoed["tool_calls"][0]["function"]["arguments"],
        json!({"name": "kitchen", "on": true})
    );
    assert_eq!(echoed["thinking"], "The user wants light.");
    assert_eq!(
        seen[1]["messages"][3],
        json!({"role": "tool", "tool_call_id": id, "tool_name": "home__set_light", "content": "kitchen is now on"})
    );
    assert_eq!(job_provider(&h).await, "ollama");
}

#[tokio::test]
async fn ollama_model_without_tools_runs_without_them() {
    let h = harness_config(
        &config_with("[llm]\nprovider = \"ollama\"\nmodel = \"gemma3:4b\"\nmax_retries = 0\n"),
        vec![
            json!({"__status": 400, "__body": {"error": "registry.ollama.ai/library/gemma3:4b does not support tools"}}),
            ollama_final("I can't control lights with this model."),
        ],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "ol-2", "double-click-hold").await;
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "retried once");
    assert!(seen[0].get("tools").is_some());
    assert!(seen[1].get("tools").is_none());
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(
        notes[0].body.contains("Unavailable: all tools"),
        "{}",
        notes[0].body
    );
    assert!(h.mcp_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_tool_arguments_are_never_run() {
    let h = harness_config(
        &config_with("[llm]\nprovider = \"openai\"\nmodel = \"gpt-5.4-mini\"\nmax_retries = 0\n"),
        vec![
            // Truncated JSON from a weak model.
            chat_tool_turn(
                "call_bad",
                "home__set_light",
                r#"{"name": "kitchen", "on": tr"#,
            ),
            chat_final("Sorry, that went wrong."),
        ],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "bad-1", "double-click-hold").await;
    assert!(h.mcp_calls.lock().unwrap().is_empty(), "never run");
    let seen = h.claude_seen.lock().unwrap().clone();
    let result = &seen[1]["messages"][3];
    assert_eq!(result["role"], "tool");
    assert_eq!(result["tool_call_id"], "call_bad");
    let text = result["content"].as_str().unwrap();
    assert!(
        text.starts_with("Error: Not run: the arguments are not valid JSON"),
        "{text}"
    );
    assert_eq!(h.notes.take()[0].body, "Sorry, that went wrong.");
}

#[tokio::test]
async fn ollama_unparseable_tool_call_fails_cleanly() {
    let h = harness_config(
        &config_with("[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\nmax_retries = 0\n"),
        vec![json!({"__status": 500, "__body": {"error": "failed to parse JSON: unexpected end of JSON input"}})],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "ol-3", "double-click-hold").await;
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Failed, "not retried for an hour");
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Error);
    assert!(
        notes[0].body.contains("malformed tool call"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn profiles_can_use_different_providers() {
    // Default (legacy [claude]) for command; capture on a local model.
    let config = CONFIG.replace(
        "system_prompt = \"Capture mode.\"",
        "system_prompt = \"Capture mode.\"\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"",
    );
    let h = harness_config(
        &config,
        vec![
            ollama_final("Added: buy milk"),
            final_turn("Kitchen lights on."),
        ],
        Some("buy milk"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "mix-1", "single-click-hold").await;
    run_command(&h, "mix-2", "double-click-hold").await;
    let m = meta(&h);
    assert_eq!(m[0]["path"], "/api/chat");
    assert_eq!(m[1]["path"], "/v1/messages");
    assert_eq!(m[1]["headers"]["x-api-key"], "sk-test");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen[0]["model"], "qwen3:8b");
    assert_eq!(seen[1]["model"], "claude-opus-5-5");
    let notes = h.notes.take();
    assert_eq!(notes[0].body, "Added: buy milk");
    assert_eq!(notes[1].body, "Kitchen lights on.");
}

#[tokio::test]
async fn state_saved_before_providers_still_resumes() {
    // A job paused for confirmation by v0.1.2 has no `provider` in its state.
    let h = harness(
        vec![
            tool_use_turn("toolu_old", "home__unlock_door", json!({"door": "side"})),
            final_turn("Side door unlocked."),
        ],
        Some("unlock the side door"),
    )
    .await;
    run_command(&h, "old-1", "double-click-hold").await;
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    {
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        let n = conn
            .execute(
                "UPDATE jobs SET agent_state = json_remove(agent_state, '$.provider') WHERE id = 1",
                [],
            )
            .unwrap();
        assert_eq!(n, 1);
        let raw: String = conn
            .query_row("SELECT agent_state FROM jobs WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(!raw.contains("\"provider\""), "{raw}");
    }
    let notes = h.notes.take();
    let Kind::Confirm {
        approve_url: Some(approve),
        ..
    } = &notes[0].kind
    else {
        panic!()
    };
    let approve_path = approve
        .strip_prefix("https://ring.test")
        .unwrap()
        .to_string();
    send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    tick(&h).await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door side".to_string()]
    );
    assert_eq!(h.notes.take().last().unwrap().body, "Side door unlocked.");
    let m = meta(&h);
    assert_eq!(m[1]["path"], "/v1/messages");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(
        seen[1]["messages"][2]["content"][0]["tool_use_id"],
        "toolu_old"
    );
    assert_eq!(job_provider(&h).await, "anthropic");
}

#[tokio::test]
async fn provider_errors_echoing_the_key_are_redacted_everywhere() {
    // A provider (or a proxy in front of it) that quotes the request's
    // credentials in its error body.
    for (config, key) in [
        (CONFIG.to_string(), "sk-test"),
        (
            config_with("[llm]\nprovider = \"openai\"\nmodel = \"gpt-6.2-sol\"\nmax_retries = 0\n"),
            "sk-openai-test",
        ),
    ] {
        let h = harness_config(
            &config,
            vec![json!({"__status": 401, "__body": {"error": {
                "type": "authentication_error",
                "message": format!("invalid credentials: x-api-key={key}, Authorization: Bearer {key}"),
            }}})],
            Some("Turn on the kitchen lights."),
            |home| vec![home],
        )
        .await;
        run_command(&h, "redact-1", "double-click-hold").await;

        let notes = h.notes.take();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].kind, Kind::Error);
        assert!(notes[0].body.contains("[redacted]"), "{}", notes[0].body);
        assert!(!format!("{notes:?}").contains(key), "{notes:?}");
        let audit = format!("{:?}", h.db.audit_entries(1).await.unwrap());
        assert!(audit.contains("[redacted]"), "{audit}");
        assert!(!audit.contains(key), "{audit}");
        let last_error: Option<String> = rusqlite::Connection::open(&h.db_path)
            .unwrap()
            .query_row("SELECT last_error FROM jobs WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert!(!last_error.unwrap_or_default().contains(key));
    }
}

#[tokio::test]
async fn missing_key_for_unused_provider_is_fine() {
    use ringmaster::config::Secrets;
    let config = Config::from_toml(&config_with(
        "[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\n",
    ))
    .unwrap();
    let secrets = Secrets {
        webhook_secret: Some(SECRET.into()),
        ..Default::default()
    };
    config.validate_runtime(&secrets).unwrap();

    // A profile on OpenAI makes its key required.
    let config = Config::from_toml(&CONFIG.replace(
        "system_prompt = \"Capture mode.\"",
        "system_prompt = \"Capture mode.\"\nprovider = \"openai\"\nmodel = \"gpt-5.4-mini\"",
    ))
    .unwrap();
    let mut secrets = Secrets {
        webhook_secret: Some(SECRET.into()),
        ..Default::default()
    };
    secrets
        .llm_keys
        .insert(Provider::Anthropic, "sk-ant".into());
    let err = config.validate_runtime(&secrets).unwrap_err().to_string();
    assert!(err.contains("OPENAI_API_KEY"), "{err}");
    secrets.llm_keys.insert(Provider::OpenAi, "sk".into());
    config.validate_runtime(&secrets).unwrap();
}

// ------------------------------------------------- one job never blocks the rest

/// The test config with extra `[agent]` keys.
fn config_agent(extra: &str) -> String {
    let anchor = "tool_discovery_timeout_secs = 2\n";
    assert!(CONFIG.contains(anchor));
    CONFIG.replace(anchor, &format!("{anchor}{extra}\n"))
}

async fn queue_command(h: &Harness, delivery: &str, phone_text: &str) {
    let (ct, body) = multipart(Some(TONE), Some(phone_text), false);
    let (status, text) = send(
        h,
        signed_request(delivery, "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
}

fn job_row(h: &Harness, id: i64) -> (String, Option<String>) {
    let conn = rusqlite::Connection::open(&h.db_path).unwrap();
    conn.query_row(
        "SELECT status, last_error FROM jobs WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
    .unwrap()
}

/// Model requests whose body contains `marker` never get an answer.
async fn hang_requests_containing(h: &Harness, marker: &str) {
    Mock::given(method("POST"))
        .and(wiremock::matchers::body_string_contains(marker))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(600)))
        .with_priority(1)
        .mount(&h._servers[1])
        .await;
}

#[tokio::test]
async fn unreadable_job_at_the_head_is_quarantined_and_the_queue_moves_on() {
    // Whisper is down, so each job's transcript is its phone transcription.
    let h = harness(vec![final_turn("Hall light on.")], None).await;
    queue_command(&h, "bad-1", "add milk to the shopping list").await;
    queue_command(&h, "good-2", "turn on the hall light").await;
    {
        // A state no version of Ringmaster can read, at the head of the queue.
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        conn.execute(
            "UPDATE jobs SET agent_state = '{\"messages\": 42, garbage' WHERE id = 1",
            [],
        )
        .unwrap();
    }

    tick(&h).await;

    let good = h.db.get_job(2).await.unwrap().unwrap();
    assert_eq!(good.status, JobStatus::Done);
    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "failed");
    let error = error.unwrap();
    assert!(error.contains("quarantined"), "{error}");
    assert!(error.contains("decoding agent state"), "{error}");

    let notes = h.notes.take();
    assert_eq!(notes.len(), 2, "{notes:?}");
    let failed = notes
        .iter()
        .find(|n| n.title == "Ringmaster: failed")
        .expect("the quarantined job is reported");
    assert_eq!(failed.kind, Kind::Error);
    assert!(failed.body.contains("could not be read"), "{}", failed.body);
    // The request is not lost: what the user said is in the notification.
    assert!(
        failed.body.contains("add milk to the shopping list"),
        "{}",
        failed.body
    );
    assert!(notes.iter().any(|n| n.body == "Hall light on."));

    let kinds: Vec<String> =
        h.db.audit_entries(1)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
    assert!(kinds.iter().any(|k| k == "job_quarantined"), "{kinds:?}");

    // Never picked up again, even after a restart's recovery sweep.
    h.db.requeue_running(now(), &[]).await.unwrap();
    tick(&h).await;
    tick(&h).await;
    assert!(h.notes.take().is_empty());
    assert_eq!(h.claude_seen.lock().unwrap().len(), 1);
    assert_eq!(job_row(&h, 1).0, "failed");
}

async fn hung_job_scenario(concurrency: usize) {
    let h = harness_config(
        &config_agent(&format!(
            "attempt_timeout_secs = 2\nworker_concurrency = {concurrency}"
        )),
        vec![final_turn("Kitchen lights on.")],
        None,
        |home| vec![home],
    )
    .await;
    hang_requests_containing(&h, "please hang").await;
    queue_command(&h, "hang-1", "please hang forever").await;
    queue_command(&h, "ok-2", "turn on the kitchen lights").await;

    // The old worker waited for the hung request (120 s) before job 2.
    tokio::time::timeout(Duration::from_secs(30), tick(&h))
        .await
        .expect("a hung job must not stall the worker");

    assert_eq!(
        h.db.get_job(2).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    // The hung attempt hit its deadline and is retried later, like an outage.
    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "queued");
    assert!(
        error.as_deref().unwrap().contains("attempt_timeout_secs"),
        "{error:?}"
    );
    let notes = h.notes.take();
    assert!(
        notes.iter().any(|n| n.body == "Kitchen lights on."),
        "{notes:?}"
    );
    assert!(
        notes
            .iter()
            .any(|n| n.title == "Ringmaster: delayed" && n.body.contains("keep trying")),
        "{notes:?}"
    );
    let audit: Vec<(String, Value)> = h.db.audit_entries(1).await.unwrap();
    assert!(
        audit.iter().any(|(k, d)| k == "retry_scheduled"
            && d["error"].as_str().unwrap().contains("no result within 2s")),
        "{audit:?}"
    );
}

#[tokio::test]
async fn hung_job_does_not_stop_others_with_concurrency() {
    hung_job_scenario(2).await;
}

#[tokio::test]
async fn hung_job_does_not_stop_others_one_at_a_time() {
    hung_job_scenario(1).await;
}

#[tokio::test]
async fn slow_job_does_not_delay_the_next_one() {
    // The deadline is far away: only concurrency lets job 2 through.
    let h = harness_config(
        &config_agent("attempt_timeout_secs = 300\nworker_concurrency = 2"),
        vec![final_turn("Kitchen lights on.")],
        None,
        |home| vec![home],
    )
    .await;
    hang_requests_containing(&h, "slow request").await;
    queue_command(&h, "slow-1", "a slow request").await;
    queue_command(&h, "fast-2", "turn on the kitchen lights").await;

    let stop = CancellationToken::new();
    let worker = tokio::spawn(h.worker.clone().run(stop.clone()));
    let done = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if h.db.get_job(2).await.unwrap().unwrap().status == JobStatus::Done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    // Still working on job 1 meanwhile.
    let slow = h.db.get_job(1).await.unwrap().unwrap().status;
    worker.abort();
    done.expect("job 2 finishes while job 1 is still running");
    assert_eq!(slow, JobStatus::Running);
}

/// Panics when a tool is called.
struct PanickingTools(Arc<dyn ToolBackend>);

#[async_trait]
impl ToolBackend for PanickingTools {
    async fn list_tools(&self, servers: &[String]) -> Result<Vec<DiscoveredTool>, ToolError> {
        self.0.list_tools(servers).await
    }
    async fn call_tool(
        &self,
        _server: &str,
        tool: &str,
        _args: serde_json::Map<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        panic!("injected panic in {tool}");
    }
    fn server_names(&self) -> Vec<String> {
        self.0.server_names()
    }
}

#[tokio::test]
async fn panic_in_one_job_fails_only_that_job() {
    let h = harness_full(
        &config_agent("worker_concurrency = 1"),
        vec![
            tool_use_turn(
                "toolu_p",
                "home__set_light",
                json!({"name": "porch", "on": true}),
            ),
            final_turn("Noted."),
        ],
        None,
        |home| vec![home],
        |tools| Arc::new(PanickingTools(tools)),
    )
    .await;
    queue_command(&h, "boom-1", "turn on the porch light").await;
    queue_command(&h, "ok-2", "remember to buy milk").await;

    tick(&h).await;

    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "failed");
    assert!(
        error
            .as_deref()
            .unwrap()
            .contains("injected panic in set_light"),
        "{error:?}"
    );
    assert_eq!(
        h.db.get_job(2).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 2, "{notes:?}");
    assert!(
        notes[0].body.contains("turn on the porch light"),
        "{notes:?}"
    );
    assert_eq!(notes[1].body, "Noted.");
    // Not retried, and not left claimed.
    tick(&h).await;
    assert_eq!(h.claude_seen.lock().unwrap().len(), 2);
    assert_eq!(job_row(&h, 1).0, "failed");
}

#[tokio::test]
async fn job_awaiting_confirmation_does_not_block_others() {
    let h = harness_config(
        &config_agent("worker_concurrency = 1"),
        vec![
            tool_use_turn("toolu_c", "home__unlock_door", json!({"door": "front"})),
            final_turn("Kitchen lights on."),
            final_turn("It is noon."),
        ],
        None,
        |home| vec![home],
    )
    .await;
    queue_command(&h, "confirm-1", "unlock the front door").await;
    queue_command(&h, "after-2", "turn on the kitchen lights").await;

    tick(&h).await;

    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    assert_eq!(
        h.db.get_job(2).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    let notes = h.notes.take();
    assert!(matches!(notes[0].kind, Kind::Confirm { .. }), "{notes:?}");
    assert_eq!(notes[1].body, "Kitchen lights on.");

    // A third request is not held up by the pending one either, and the
    // pending job is not claimed while it waits.
    queue_command(&h, "later-3", "what time is it").await;
    tick(&h).await;
    assert_eq!(
        h.db.get_job(3).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    assert_eq!(h.notes.take()[0].body, "It is noon.");
    assert_eq!(h.claude_seen.lock().unwrap().len(), 3);
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    assert!(h.mcp_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn started_job_is_not_resumed_long_after_its_max_age() {
    let h = harness(
        vec![
            json!({"__status": 503, "__body": {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}}),
            final_turn("Heating off."),
        ],
        None,
    )
    .await;
    queue_command(&h, "late-1", "turn the heating off").await;
    tick(&h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Queued);
    assert!(job.agent_state.is_some(), "the conversation was started");
    h.notes.take();

    // The service was down (or kept crashing on this job) for a day; the
    // job comes back through the restart sweep, not a scheduled retry.
    {
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        let day_ago = now() - 86_400;
        conn.execute(
            "UPDATE jobs SET status = 'running', created_at = ?1 WHERE id = 1",
            [day_ago],
        )
        .unwrap();
        conn.execute(
            "UPDATE deliveries SET recorded_at_ms = ?1",
            [day_ago * 1000],
        )
        .unwrap();
    }
    tick(&h).await;

    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "failed");
    assert!(
        error.as_deref().unwrap().contains("too old to resume"),
        "{error:?}"
    );
    assert_eq!(h.claude_seen.lock().unwrap().len(), 1, "not resumed");
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    assert!(notes[0].body.contains("turn the heating off"), "{notes:?}");
}
