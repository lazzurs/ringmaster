//! The agent loop: Claude Messages API + MCP tools, with confirmations.
//!
//! The conversation is stored in the job after every step, so a job can be
//! retried, survive a restart, or pause for an Approve/Deny decision and pick
//! up exactly where it left off. History is append-only: the system prompt,
//! model and tool list are frozen when the conversation starts, and assistant
//! content (thinking blocks included) is echoed back unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::claude::{ClaudeClient, ClaudeError, FALLBACK_BETA};
use crate::config::{Config, ProfileConfig};
use crate::db::{Db, Job, JobStatus};
use crate::notify::{Kind, Notification, Notifier};
use crate::policy::{Policies, Policy, ToolFilter};
use crate::tools::{ToolBackend, ToolDef, ToolError, to_defs};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentState {
    pub model: String,
    pub system: String,
    pub effort: String,
    pub tools: Vec<ToolDef>,
    pub messages: Vec<Value>,
    pub iterations: u32,
    pub pending: Option<PendingTurn>,
    /// Set once Claude has given its final answer.
    #[serde(default)]
    pub final_text: Option<String>,
}

/// Tool calls of one assistant turn, held until the user decides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTurn {
    pub calls: Vec<PlannedCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedCall {
    pub tool_use_id: String,
    pub claude_name: String,
    pub input: Value,
    pub action: CallAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "reason", rename_all = "snake_case")]
pub enum CallAction {
    Run,
    Confirm,
    Reject(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentOutcome {
    /// Finished; the text is the reply for the user.
    Final(String),
    /// Paused for an Approve/Deny decision.
    AwaitingConfirmation,
    /// Claude (or its safety classifiers) declined.
    Refused(String),
    /// Stopped without an answer (iteration cap).
    GaveUp(String),
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("{0}")]
    Transient(String),
    #[error("{0}")]
    Permanent(String),
}

impl From<anyhow::Error> for AgentError {
    fn from(e: anyhow::Error) -> Self {
        // Local failures (database, I/O) are worth another try.
        AgentError::Transient(format!("{e:#}"))
    }
}

/// What the agent knows about the request.
pub struct Request<'a> {
    pub transcript: &'a str,
    pub transcript_source: &'a str,
}

pub struct Agent {
    pub config: Arc<Config>,
    pub db: Db,
    pub claude: ClaudeClient,
    pub tools: Arc<dyn ToolBackend>,
    pub notifier: Arc<dyn Notifier>,
    policies: Policies,
    filters: HashMap<String, ToolFilter>,
}

impl Agent {
    pub fn new(
        config: Arc<Config>,
        db: Db,
        claude: ClaudeClient,
        tools: Arc<dyn ToolBackend>,
        notifier: Arc<dyn Notifier>,
    ) -> anyhow::Result<Self> {
        let policies = Policies::new(&config.tool_policies)?;
        let filters = config
            .profiles
            .iter()
            .map(|(n, p)| Ok((n.clone(), ToolFilter::new(&p.tools)?)))
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            config,
            db,
            claude,
            tools,
            notifier,
            policies,
            filters,
        })
    }

    fn profile(&self, name: &str) -> Result<(&ProfileConfig, &ToolFilter), AgentError> {
        match (self.config.profiles.get(name), self.filters.get(name)) {
            (Some(p), Some(f)) => Ok((p, f)),
            _ => Err(AgentError::Permanent(format!(
                "profile {name:?} no longer exists"
            ))),
        }
    }

    pub async fn run(&self, job: &Job, req: &Request<'_>) -> Result<AgentOutcome, AgentError> {
        let (profile, filter) = self.profile(&job.profile)?;
        let mut state = job.agent_state.clone().unwrap_or_default();
        if let Some(text) = &state.final_text {
            // Finished before a restart; only the bookkeeping was lost.
            return Ok(AgentOutcome::Final(text.clone()));
        }

        if state.messages.is_empty() {
            self.start(job, req, profile, filter, &mut state).await?;
        }

        if let Some(pending) = state.pending.clone() {
            let decision = self.db.latest_decision(job.id).await?;
            let Some(decision) = decision else {
                // Still waiting (should not have been queued).
                self.db
                    .set_status(job.id, JobStatus::AwaitingConfirmation)
                    .await?;
                return Ok(AgentOutcome::AwaitingConfirmation);
            };
            let approved = decision == "approve";
            self.audit(
                job,
                "confirmation_decision",
                json!({ "decision": decision }),
            )
            .await;
            let refusal_text = if decision == "expired" {
                "Not run: the user did not answer the confirmation request in time."
            } else {
                "Not run: the user declined this action."
            };
            let results = self
                .execute(job, &state, &pending.calls, approved, refusal_text)
                .await;
            state
                .messages
                .push(json!({"role": "user", "content": results}));
            state.pending = None;
            self.db.save_agent_state(job.id, &state).await?;
        }

        let max_iterations = profile
            .max_iterations
            .unwrap_or(self.config.claude.max_iterations);

        loop {
            // A crash between storing Claude's reply and its tool results
            // leaves an unanswered assistant turn: answer it now.
            if last_role(&state.messages) == Some("assistant")
                && has_tool_use(state.messages.last())
            {
                if let Some(outcome) = self.handle_tool_turn(job, req, filter, &mut state).await? {
                    return Ok(outcome);
                }
                continue;
            }

            if state.iterations >= max_iterations {
                return Ok(AgentOutcome::GaveUp(format!(
                    "I stopped after {max_iterations} steps without finishing."
                )));
            }

            let (body, betas) = self.request_body(&state);
            state.iterations += 1;
            let resp = match self.claude.create_message(&body, &betas).await {
                Ok(r) => r,
                Err(ClaudeError::Transient(e)) => {
                    state.iterations -= 1;
                    return Err(AgentError::Transient(e));
                }
                Err(ClaudeError::Permanent(e)) => {
                    self.audit(job, "claude_error", json!({ "error": e })).await;
                    return Err(AgentError::Permanent(e));
                }
            };

            let stop_reason = resp
                .get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let content = sanitize_fallback(resp.get("content").cloned().unwrap_or(json!([])));
            self.audit(job, "claude_response", summarize_response(&resp, &content))
                .await;

            match stop_reason.as_str() {
                "refusal" => {
                    let category = resp
                        .pointer("/stop_details/category")
                        .and_then(Value::as_str)
                        .unwrap_or("unspecified");
                    self.db.save_agent_state(job.id, &state).await?;
                    return Ok(AgentOutcome::Refused(format!(
                        "Claude declined this request (category: {category})."
                    )));
                }
                "max_tokens" => {
                    // A truncated tool_use input may still parse; never run it.
                    self.db.save_agent_state(job.id, &state).await?;
                    return Err(AgentError::Permanent(
                        "Claude's reply was cut off at max_tokens; raise claude.max_tokens".into(),
                    ));
                }
                "pause_turn" => {
                    state
                        .messages
                        .push(json!({"role": "assistant", "content": content}));
                    self.db.save_agent_state(job.id, &state).await?;
                    continue;
                }
                "tool_use" => {
                    state
                        .messages
                        .push(json!({"role": "assistant", "content": content}));
                    self.db.save_agent_state(job.id, &state).await?;
                    if let Some(outcome) =
                        self.handle_tool_turn(job, req, filter, &mut state).await?
                    {
                        return Ok(outcome);
                    }
                }
                _ => {
                    // end_turn, stop_sequence, or anything new: the answer.
                    let text = final_text(&content);
                    state
                        .messages
                        .push(json!({"role": "assistant", "content": content}));
                    state.final_text = Some(text.clone());
                    self.db.save_agent_state(job.id, &state).await?;
                    return Ok(AgentOutcome::Final(text));
                }
            }
        }
    }

    async fn start(
        &self,
        job: &Job,
        req: &Request<'_>,
        profile: &ProfileConfig,
        filter: &ToolFilter,
        state: &mut AgentState,
    ) -> Result<(), AgentError> {
        let servers: Vec<String> = self
            .tools
            .server_names()
            .into_iter()
            .filter(|s| filter.may_use_server(s))
            .collect();
        // Each server on its own: a server the profile names explicitly must
        // be up (the job retries later), but one reached only through a
        // wildcard is skipped while it is down, so one broken integration
        // does not stop every command.
        let mut discovered = Vec::new();
        let mut unavailable = Vec::new();
        for server in &servers {
            match self.tools.list_tools(std::slice::from_ref(server)).await {
                Ok(tools) => discovered.extend(tools),
                Err(e) if filter.requires_server(server) => {
                    return Err(AgentError::Transient(match e {
                        ToolError::Unreachable(..) => e.to_string(),
                        ToolError::Other(m) => m,
                    }));
                }
                Err(e) => {
                    tracing::warn!(job = job.id, server = %server, error = %e, "skipping unavailable tool server");
                    self.audit(
                        job,
                        "tool_server_skipped",
                        json!({"server": server, "error": e.to_string()}),
                    )
                    .await;
                    unavailable.push(server.clone());
                }
            }
        }
        let offered: Vec<_> = discovered
            .into_iter()
            .filter(|t| {
                let q = t.qualified();
                filter.allows(&q) && self.policies.policy_for(&q) != Policy::Deny
            })
            .collect();
        state.tools = to_defs(offered);
        state.model = profile
            .model
            .clone()
            .unwrap_or_else(|| self.config.claude.model.clone());
        state.effort = profile
            .effort
            .clone()
            .unwrap_or_else(|| self.config.claude.effort.clone());
        state.system = format!(
            "{}\n\n{}",
            self.config.base_prompt().trim(),
            profile.system_prompt.trim()
        );
        state.messages = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": self.context_text(job, req, &unavailable)}],
        })];
        self.audit(
            job,
            "agent_start",
            json!({
                "model": state.model,
                "effort": state.effort,
                "tools": state.tools.iter().map(ToolDef::qualified).collect::<Vec<_>>(),
            }),
        )
        .await;
        self.db.save_agent_state(job.id, state).await?;
        Ok(())
    }

    /// Volatile context goes in the first user message, never in the system
    /// prompt, so the cached prefix (tools + system) stays identical.
    fn context_text(&self, job: &Job, req: &Request<'_>, unavailable: &[String]) -> String {
        let tz =
            jiff::tz::TimeZone::get(&self.config.agent.timezone).unwrap_or(jiff::tz::TimeZone::UTC);
        let fmt = |ts: jiff::Timestamp| {
            ts.to_zoned(tz.clone())
                .strftime("%A %Y-%m-%d %H:%M:%S %Z")
                .to_string()
        };
        let now = fmt(jiff::Timestamp::now());
        let recorded = job
            .recorded_at_ms
            .and_then(|ms| jiff::Timestamp::from_millisecond(ms).ok())
            .map(fmt)
            .unwrap_or_else(|| "unknown".into());
        let unavailable = if unavailable.is_empty() {
            String::new()
        } else {
            format!(
                "Unavailable right now (their tools are missing; say so if the request needs them): {}\n",
                unavailable.join(", ")
            )
        };
        format!(
            "<context>\nCurrent time: {now} ({tzname})\nRecorded at: {recorded}\nGesture: {trigger}\nTranscript source: {source}\n{unavailable}</context>\n<transcript>\n{text}\n</transcript>",
            tzname = self.config.agent.timezone,
            trigger = job.trigger,
            source = req.transcript_source,
            text = req.transcript.trim(),
        )
    }

    fn request_body(&self, state: &AgentState) -> (Value, Vec<&'static str>) {
        let mut body = json!({
            "model": state.model,
            "max_tokens": self.config.claude.max_tokens,
            "system": [{
                "type": "text",
                "text": state.system,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": state.messages,
            "output_config": {"effort": state.effort},
            // Automatic breakpoint for the growing conversation tail.
            "cache_control": {"type": "ephemeral"},
        });
        if !state.tools.is_empty() {
            body["tools"] = Value::Array(state.tools.iter().map(ToolDef::to_api).collect());
        }
        let mut betas = Vec::new();
        if self.config.claude.fallbacks_enabled(&state.model) {
            body["fallbacks"] = json!("default");
            betas.push(FALLBACK_BETA);
        }
        (body, betas)
    }

    /// Plan and run (or park) the tool calls of the last assistant turn.
    /// Returns `Some` when the loop must stop.
    async fn handle_tool_turn(
        &self,
        job: &Job,
        req: &Request<'_>,
        filter: &ToolFilter,
        state: &mut AgentState,
    ) -> Result<Option<AgentOutcome>, AgentError> {
        let content = state
            .messages
            .last()
            .and_then(|m| m.get("content"))
            .cloned()
            .unwrap_or(json!([]));
        let uses: Vec<&Value> = content
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                    .collect()
            })
            .unwrap_or_default();
        if uses.is_empty() {
            return Ok(Some(AgentOutcome::Final(final_text(&content))));
        }

        let calls: Vec<PlannedCall> = uses
            .iter()
            .map(|u| {
                let id = u
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = u
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let input = u.get("input").cloned().unwrap_or(json!({}));
                let action = match state.tools.iter().find(|d| d.claude_name == name) {
                    None => CallAction::Reject(format!("Unknown tool {name}.")),
                    Some(def) => {
                        let q = def.qualified();
                        if !filter.allows(&q) {
                            CallAction::Reject(
                                "This tool is not available for this gesture.".into(),
                            )
                        } else {
                            match self.policies.policy_for(&q) {
                                Policy::Allow => CallAction::Run,
                                Policy::Confirm => CallAction::Confirm,
                                Policy::Deny => {
                                    CallAction::Reject("This tool is blocked by policy.".into())
                                }
                            }
                        }
                    }
                };
                PlannedCall {
                    tool_use_id: id,
                    claude_name: name,
                    input,
                    action,
                }
            })
            .collect();

        if calls.iter().any(|c| c.action == CallAction::Confirm) {
            state.pending = Some(PendingTurn {
                calls: calls.clone(),
            });
            self.db.save_agent_state(job.id, state).await?;
            self.request_confirmation(job, req, state, &calls).await?;
            return Ok(Some(AgentOutcome::AwaitingConfirmation));
        }

        let results = self.execute(job, state, &calls, false, "").await;
        state
            .messages
            .push(json!({"role": "user", "content": results}));
        self.db.save_agent_state(job.id, state).await?;
        Ok(None)
    }

    async fn request_confirmation(
        &self,
        job: &Job,
        req: &Request<'_>,
        state: &AgentState,
        calls: &[PlannedCall],
    ) -> Result<(), AgentError> {
        let token_bytes: [u8; 32] = rand::random();
        let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(token_bytes);
        let token_hash = hash_token(&token);
        let lines: Vec<String> = calls
            .iter()
            .filter(|c| c.action == CallAction::Confirm)
            .map(|c| {
                let q = state
                    .tools
                    .iter()
                    .find(|d| d.claude_name == c.claude_name)
                    .map(ToolDef::qualified)
                    .unwrap_or_else(|| c.claude_name.clone());
                format!("{q} {}", truncate(&c.input.to_string(), 300))
            })
            .collect();
        let summary = format!(
            "Request: \"{}\"\nRun {}:\n{}",
            truncate(req.transcript.trim(), 200),
            if lines.len() == 1 {
                "this action"
            } else {
                "these actions"
            },
            lines.join("\n")
        );
        let ttl = self.config.agent.confirmation_ttl_secs as i64;
        let expires_at = crate::db::now() + ttl;
        self.db
            .create_confirmation(token_hash, job.id, summary.clone(), expires_at)
            .await?;
        self.db
            .set_status(job.id, JobStatus::AwaitingConfirmation)
            .await?;
        self.audit(
            job,
            "confirmation_requested",
            json!({ "summary": summary, "expires_at": expires_at }),
        )
        .await;
        let base = self
            .config
            .server
            .public_base_url
            .as_deref()
            .map(|b| b.trim_end_matches('/').to_string());
        let kind = Kind::Confirm {
            approve_url: base
                .as_ref()
                .map(|b| format!("{b}/confirm/{token}/approve")),
            deny_url: base.as_ref().map(|b| format!("{b}/confirm/{token}/deny")),
            view_url: base.as_ref().map(|b| format!("{b}/confirm/{token}")),
        };
        let n = Notification {
            title: "Ringmaster: approve?".into(),
            body: format!("{summary}\n(expires in {} min)", ttl / 60),
            kind,
        };
        if let Err(e) = self.notifier.send(&n).await {
            tracing::error!(job = job.id, error = %e, "failed to send confirmation notification");
        }
        Ok(())
    }

    /// Run planned calls in order and build the `tool_result` blocks, all of
    /// which go back to Claude in one user message.
    async fn execute(
        &self,
        job: &Job,
        state: &AgentState,
        calls: &[PlannedCall],
        approved: bool,
        refusal_text: &str,
    ) -> Vec<Value> {
        let max = self.config.agent.max_tool_result_chars;
        let mut results = Vec::with_capacity(calls.len());
        for c in calls {
            let def = state.tools.iter().find(|d| d.claude_name == c.claude_name);
            let qualified = def
                .map(ToolDef::qualified)
                .unwrap_or_else(|| c.claude_name.clone());
            let run = match &c.action {
                CallAction::Run => Ok(()),
                CallAction::Confirm if approved => Ok(()),
                CallAction::Confirm => Err(refusal_text.to_string()),
                CallAction::Reject(r) => Err(r.clone()),
            };
            let (text, is_error, outcome) = match (run, def) {
                (Err(reason), _) => (reason, true, "not_run"),
                (Ok(()), None) => ("Unknown tool.".to_string(), true, "not_run"),
                (Ok(()), Some(def)) => {
                    let args = c.input.as_object().cloned().unwrap_or_default();
                    match self.tools.call_tool(&def.server, &def.tool, args).await {
                        Ok(out) => (
                            out.text,
                            out.is_error,
                            if out.is_error { "tool_error" } else { "ok" },
                        ),
                        Err(e) => (format!("Tool call failed: {e}"), true, "call_failed"),
                    }
                }
            };
            metrics::counter!("ringmaster_tool_calls_total", "outcome" => outcome).increment(1);
            self.audit(
                job,
                "tool_call",
                json!({
                    "tool": qualified,
                    "input": c.input,
                    "action": c.action,
                    "approved": approved,
                    "outcome": outcome,
                    "result": truncate(&text, 2000),
                }),
            )
            .await;
            let mut block = json!({
                "type": "tool_result",
                "tool_use_id": c.tool_use_id,
                "content": truncate(&text, max),
            });
            if is_error {
                block["is_error"] = json!(true);
            }
            results.push(block);
        }
        results
    }

    async fn audit(&self, job: &Job, kind: &str, detail: Value) {
        if let Err(e) = self
            .db
            .audit(Some(job.id), Some(&job.delivery_id), kind, detail)
            .await
        {
            tracing::error!(job = job.id, error = %e, "audit write failed");
        }
    }
}

pub fn hash_token(token: &str) -> String {
    hex::encode(Sha256::digest(token.as_bytes()))
}

fn has_tool_use(message: Option<&Value>) -> bool {
    message
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .is_some_and(|a| {
            a.iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
        })
}

fn last_role(messages: &[Value]) -> Option<&str> {
    messages
        .last()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
}

/// After a server-side fallback, blocks of the declined attempt that come
/// before the last `fallback` block must not be echoed back (thinking,
/// redacted thinking, tool use). The `fallback` block itself stays in place.
pub fn sanitize_fallback(content: Value) -> Value {
    let Value::Array(blocks) = content else {
        return content;
    };
    let Some(last_fb) = blocks
        .iter()
        .rposition(|b| b.get("type").and_then(Value::as_str) == Some("fallback"))
    else {
        return Value::Array(blocks);
    };
    Value::Array(
        blocks
            .into_iter()
            .enumerate()
            .filter(|(i, b)| {
                *i >= last_fb
                    || !matches!(
                        b.get("type").and_then(Value::as_str),
                        Some("thinking" | "redacted_thinking" | "tool_use" | "server_tool_use")
                    )
            })
            .map(|(_, b)| b)
            .collect(),
    )
}

fn final_text(content: &Value) -> String {
    let text = content
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let text = text.trim();
    if text.is_empty() {
        "Done.".into()
    } else {
        text.to_string()
    }
}

fn summarize_response(resp: &Value, content: &Value) -> Value {
    let blocks: Vec<&str> = content
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|b| b.get("type").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    let tool_uses: Vec<Value> = content
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                .map(|b| json!({"name": b.get("name"), "input": b.get("input")}))
                .collect()
        })
        .unwrap_or_default();
    let fallback_served = resp
        .pointer("/usage/iterations")
        .and_then(Value::as_array)
        .is_some_and(|its| {
            its.iter()
                .any(|i| i.get("type").and_then(Value::as_str) == Some("fallback_message"))
        });
    json!({
        "id": resp.get("id"),
        "model": resp.get("model"),
        "stop_reason": resp.get("stop_reason"),
        "stop_details": resp.get("stop_details"),
        "fallback_served": fallback_served,
        "blocks": blocks,
        "tool_uses": tool_uses,
        "text": truncate(&final_text(content), 2000),
        "usage": resp.get("usage").map(|u| json!({
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        })),
    })
}

pub fn truncate(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}… [truncated]", &s[..i]),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_sanitizer_drops_declined_attempt_blocks() {
        let content = json!([
            {"type": "thinking", "thinking": "", "signature": "a"},
            {"type": "tool_use", "id": "t0", "name": "x", "input": {}},
            {"type": "text", "text": "partial"},
            {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"}},
            {"type": "thinking", "thinking": "", "signature": "b"},
            {"type": "tool_use", "id": "t1", "name": "y", "input": {}}
        ]);
        let out = sanitize_fallback(content);
        let types: Vec<_> = out
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(types, ["text", "fallback", "thinking", "tool_use"]);
        assert_eq!(out[3]["id"], "t1");

        let plain = json!([{"type": "thinking", "thinking": "", "signature": "a"}, {"type": "text", "text": "hi"}]);
        assert_eq!(sanitize_fallback(plain.clone()), plain);
    }

    #[test]
    fn final_text_reads_text_blocks_only() {
        let c = json!([{"type": "thinking", "thinking": "x"}, {"type": "text", "text": " Lights on. "}]);
        assert_eq!(final_text(&c), "Lights on.");
        assert_eq!(final_text(&json!([])), "Done.");
    }

    #[test]
    fn truncate_marks_cut() {
        assert_eq!(truncate("abc", 5), "abc");
        assert_eq!(truncate("abcdef", 3), "abc… [truncated]");
    }
}
