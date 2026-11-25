//! The agent loop: an LLM (Claude or another provider) + MCP tools, with
//! confirmations.
//!
//! The conversation is stored in the job after every step, so a job can be
//! retried, survive a restart, or pause for an Approve/Deny decision and pick
//! up exactly where it left off. History is append-only: the provider, model,
//! system prompt and tool list are frozen when the conversation starts, and
//! assistant messages (Claude's thinking blocks included) are echoed back
//! unchanged in the provider's own format.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::config::{Config, ProfileConfig};
use crate::db::{Db, Job, JobStatus};
use crate::llm::{ChatModel, ChatRequest, LlmError, Llms, Provider, StopKind, ToolResult};
use crate::notify::{Kind, Notification, Notifier};
use crate::policy::{Policies, Policy, ToolFilter};
use crate::tools::{ToolBackend, ToolDef, ToolError, to_defs};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentState {
    /// Provider the job started with; it keeps it for good. States saved
    /// before there was a choice are Anthropic's.
    #[serde(default)]
    pub provider: Provider,
    pub model: String,
    pub system: String,
    pub effort: String,
    pub tools: Vec<ToolDef>,
    pub messages: Vec<Value>,
    pub iterations: u32,
    pub pending: Option<PendingTurn>,
    /// Set once the model has given its final answer.
    #[serde(default)]
    pub final_text: Option<String>,
    /// MCP servers (or "all tools") that could not be used for this job.
    #[serde(default)]
    pub unavailable: Vec<String>,
}

/// Tool calls of one assistant turn, held until the user decides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTurn {
    pub calls: Vec<PlannedCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedCall {
    /// The provider's id for the call (`tool_use_id` / `tool_call_id`).
    pub tool_use_id: String,
    /// The tool name the model used.
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
    /// The model (or its provider's safety systems) declined.
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
    pub llms: Llms,
    pub tools: Arc<dyn ToolBackend>,
    pub notifier: Arc<dyn Notifier>,
    policies: Policies,
    filters: HashMap<String, ToolFilter>,
}

impl Agent {
    pub fn new(
        config: Arc<Config>,
        db: Db,
        llms: Llms,
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
            llms,
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
        let llm = self.llms.get(state.provider);

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
            state.messages.extend(llm.tool_results(&results));
            state.pending = None;
            self.db.save_agent_state(job.id, &state).await?;
        }

        let max_iterations = self.config.max_iterations(profile);
        let (response_kind, error_kind) = llm.audit_kinds();

        loop {
            // A crash between storing the model's reply and its tool results
            // leaves an unanswered assistant turn: answer it now.
            if let Some(last) = state.messages.last()
                && last.get("role").and_then(Value::as_str) == Some("assistant")
                && !llm.tool_calls(last).is_empty()
            {
                if let Some(outcome) = self
                    .handle_tool_turn(job, req, filter, llm, &mut state)
                    .await?
                {
                    return Ok(outcome);
                }
                continue;
            }

            if state.iterations >= max_iterations {
                return Ok(AgentOutcome::GaveUp(format!(
                    "I stopped after {max_iterations} steps without finishing."
                )));
            }

            state.iterations += 1;
            let result = llm
                .complete(&ChatRequest {
                    model: &state.model,
                    system: &state.system,
                    effort: &state.effort,
                    tools: &state.tools,
                    messages: &state.messages,
                })
                .await;
            let turn = match result {
                Ok(t) => t,
                Err(LlmError::Transient(e)) => {
                    state.iterations -= 1;
                    return Err(AgentError::Transient(e));
                }
                Err(LlmError::Permanent(e)) => {
                    self.audit(job, error_kind, json!({ "error": e })).await;
                    // A tool definition the API rejects (or a model that
                    // cannot use tools at all) must not sink the request.
                    // Before any tool has run, try once more with no tools:
                    // the model can still answer, and say what it could not
                    // do.
                    if !state.tools.is_empty() && state.messages.len() == 1 && rejects_tools(&e) {
                        tracing::warn!(job = job.id, provider = %state.provider, error = %e, "API rejected the request with tools; retrying without them");
                        self.audit(
                            job,
                            "tools_dropped",
                            json!({
                                "tools": state.tools.iter().map(ToolDef::qualified).collect::<Vec<_>>(),
                                "error": e,
                            }),
                        )
                        .await;
                        state.tools.clear();
                        state.unavailable.push("all tools".into());
                        state.iterations -= 1;
                        self.db.save_agent_state(job.id, &state).await?;
                        continue;
                    }
                    return Err(AgentError::Permanent(e));
                }
            };
            self.audit(job, response_kind, turn.audit.clone()).await;

            match turn.stop {
                StopKind::Refusal(msg) => {
                    self.db.save_agent_state(job.id, &state).await?;
                    return Ok(AgentOutcome::Refused(msg));
                }
                StopKind::MaxTokens => {
                    // A truncated tool call may still parse; never run it.
                    self.db.save_agent_state(job.id, &state).await?;
                    return Err(AgentError::Permanent(format!(
                        "{}'s reply was cut off at max_tokens; raise max_tokens for {}",
                        state.provider.display(),
                        state.provider
                    )));
                }
                StopKind::Pause => {
                    state.messages.push(turn.message);
                    self.db.save_agent_state(job.id, &state).await?;
                    continue;
                }
                StopKind::ToolUse => {
                    state.messages.push(turn.message);
                    self.db.save_agent_state(job.id, &state).await?;
                    if let Some(outcome) = self
                        .handle_tool_turn(job, req, filter, llm, &mut state)
                        .await?
                    {
                        return Ok(outcome);
                    }
                }
                StopKind::EndTurn => {
                    let text = with_unavailable_note(llm.final_text(&turn.message), &state, req);
                    state.messages.push(turn.message);
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
        // Every server is optional: one that is down, slow, or misbehaving is
        // skipped and the job runs with the rest. Servers are listed
        // concurrently with a short timeout so a hung one costs seconds, not
        // minutes, and never blocks the others.
        let timeout = std::time::Duration::from_secs(self.config.agent.tool_discovery_timeout_secs);
        let listings = futures_util::future::join_all(servers.iter().map(|server| {
            let tools = self.tools.clone();
            async move {
                let res =
                    tokio::time::timeout(timeout, tools.list_tools(std::slice::from_ref(server)))
                        .await
                        .unwrap_or_else(|_| {
                            Err(ToolError::Unreachable(
                                server.clone(),
                                format!("no answer within {}s", timeout.as_secs()),
                            ))
                        });
                (server.clone(), res)
            }
        }))
        .await;
        let mut discovered = Vec::new();
        let mut unavailable = Vec::new();
        for (server, res) in listings {
            match res {
                Ok(tools) => {
                    for t in tools {
                        match crate::tools::check_schema(&t.input_schema) {
                            Ok(()) => discovered.push(t),
                            Err(why) => {
                                tracing::warn!(job = job.id, tool = %t.qualified(), %why, "skipping tool with an unusable schema");
                                self.audit(
                                    job,
                                    "tool_skipped",
                                    json!({"tool": t.qualified(), "reason": why}),
                                )
                                .await;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(job = job.id, server = %server, error = %e, "skipping unavailable tool server");
                    metrics::counter!("ringmaster_tool_server_skipped_total", "server" => server.clone()).increment(1);
                    self.audit(
                        job,
                        "tool_server_skipped",
                        json!({"server": server, "error": e.to_string()}),
                    )
                    .await;
                    unavailable.push(server);
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
        let choice = self
            .config
            .profile_llm(Some(profile))
            .map_err(|e| AgentError::Permanent(format!("{e:#}")))?;
        let llm = self.llms.get(choice.provider);
        state.tools = to_defs(offered);
        state.unavailable = unavailable.clone();
        state.provider = choice.provider;
        state.model = choice.model;
        state.effort = choice.effort;
        state.system = format!(
            "{}\n\n{}",
            self.config.base_prompt().trim(),
            profile.system_prompt.trim()
        );
        state.messages = vec![llm.user_message(&self.context_text(job, req, &unavailable))];
        self.audit(
            job,
            "agent_start",
            json!({
                "provider": state.provider,
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

    /// Plan and run (or park) the tool calls of the last assistant turn.
    /// Returns `Some` when the loop must stop.
    async fn handle_tool_turn(
        &self,
        job: &Job,
        req: &Request<'_>,
        filter: &ToolFilter,
        llm: &dyn ChatModel,
        state: &mut AgentState,
    ) -> Result<Option<AgentOutcome>, AgentError> {
        let last = state.messages.last().cloned().unwrap_or(Value::Null);
        let uses = llm.tool_calls(&last);
        if uses.is_empty() {
            return Ok(Some(AgentOutcome::Final(with_unavailable_note(
                llm.final_text(&last),
                state,
                req,
            ))));
        }

        let calls: Vec<PlannedCall> = uses
            .into_iter()
            .map(|u| {
                let name = u.name;
                let (input, bad_input) = match u.input {
                    Ok(v) => (v, None),
                    Err(why) => (Value::Null, Some(why)),
                };
                let action = match state.tools.iter().find(|d| d.claude_name == name) {
                    // Malformed arguments (a small model, or a cut-off reply)
                    // are never run, whatever the tool's policy.
                    _ if bad_input.is_some() => CallAction::Reject(format!(
                        "Not run: {}. Call the tool again with a JSON object as its arguments.",
                        bad_input.as_deref().unwrap_or_default()
                    )),
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
                    tool_use_id: u.id,
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
        state.messages.extend(llm.tool_results(&results));
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

    /// Run planned calls in order; the results go back to the model in the
    /// provider's format.
    async fn execute(
        &self,
        job: &Job,
        state: &AgentState,
        calls: &[PlannedCall],
        approved: bool,
        refusal_text: &str,
    ) -> Vec<ToolResult> {
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
            results.push(ToolResult {
                id: c.tool_use_id.clone(),
                name: c.claude_name.clone(),
                text: truncate(&text, max),
                is_error,
            });
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

/// Whether a permanent error is worth one more try without tools: a 400
/// that is not about something tools cannot fix (thinking/reasoning).
fn rejects_tools(e: &str) -> bool {
    let lower = e.to_ascii_lowercase();
    e.starts_with("400") && !lower.contains("thinking") && !lower.contains("reasoning")
}

/// Kept for callers of the Claude-only API.
pub use crate::llm::anthropic::sanitize_fallback;

/// When some tools were unavailable, end the reply with which ones and what
/// the user said, so a request that could not be carried out is never lost.
fn with_unavailable_note(text: String, state: &AgentState, req: &Request<'_>) -> String {
    if state.unavailable.is_empty() {
        return text;
    }
    format!(
        "{}\n\nUnavailable: {}. You said: \"{}\"",
        text.trim_end(),
        state.unavailable.join(", "),
        truncate(req.transcript.trim(), 500)
    )
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
    fn old_states_without_provider_are_anthropic() {
        let old = json!({
            "model": "claude-opus-5-5", "system": "s", "effort": "low", "tools": [],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "iterations": 1,
            "pending": {"calls": [{"tool_use_id": "toolu_1", "claude_name": "home__x",
                                   "input": {}, "action": {"kind": "confirm"}}]}
        });
        let s: AgentState = serde_json::from_value(old).unwrap();
        assert_eq!(s.provider, Provider::Anthropic);
        assert_eq!(s.pending.unwrap().calls[0].action, CallAction::Confirm);
        let new = serde_json::to_value(AgentState {
            provider: Provider::Ollama,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(new["provider"], "ollama");
    }

    #[test]
    fn tools_dropped_only_for_plain_400s() {
        assert!(rejects_tools(
            "400 Bad Request: invalid_request_error: tools.0: bad"
        ));
        assert!(rejects_tools(
            "400 Bad Request: registry.ollama.ai/library/gemma3:4b does not support tools"
        ));
        assert!(!rejects_tools(
            "400 Bad Request: \"gemma3\" does not support thinking"
        ));
        assert!(!rejects_tools("401 Unauthorized: bad key"));
    }

    #[test]
    fn truncate_marks_cut() {
        assert_eq!(truncate("abc", 5), "abc");
        assert_eq!(truncate("abcdef", 3), "abc… [truncated]");
    }
}
