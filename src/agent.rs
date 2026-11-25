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

use crate::classifier::{Classifier, ClassifyError, ProposedCall, Verdict};
use crate::config::{ApprovalMode, Config, ProfileConfig};
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
    /// Calls the approval classifier let run without asking; listed in the
    /// final notification.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auto_approved: Vec<AutoApproval>,
}

/// A `confirm` call that ran on the classifier's approval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoApproval {
    /// `<server>/<tool>`.
    pub tool: String,
    pub input: Value,
    pub score: f64,
}

/// Tool calls of one assistant turn, held until the user decides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTurn {
    pub calls: Vec<PlannedCall>,
    /// The approval mode the turn was judged in (unset: manual).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<ApprovalMode>,
    /// The classifier's say on each `confirm` call, in shadow and
    /// classifier mode.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub judgements: Vec<Judgement>,
}

/// What the approval step made of one `confirm` call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Judgement {
    pub tool_use_id: String,
    /// `<server>/<tool>`.
    pub tool: String,
    pub outcome: Judged,
    /// The classifier's score, when it gave one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Why there is no score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Judged {
    Safe,
    Uncertain,
    Unsafe,
    /// No verdict (down, timed out, malformed reply, too long to judge).
    Unavailable,
    /// Policy `ask`: always a human, never classified.
    Ask,
    /// Not classified: another call in the turn needs a human anyway.
    Skipped,
}

impl Judged {
    pub fn as_str(self) -> &'static str {
        match self {
            Judged::Safe => "safe",
            Judged::Uncertain => "uncertain",
            Judged::Unsafe => "unsafe",
            Judged::Unavailable => "unavailable",
            Judged::Ask => "ask",
            Judged::Skipped => "skipped",
        }
    }
}

impl From<Verdict> for Judged {
    fn from(v: Verdict) -> Self {
        match v {
            Verdict::Safe => Judged::Safe,
            Verdict::Uncertain => Judged::Uncertain,
            Verdict::Unsafe => Judged::Unsafe,
        }
    }
}

/// Who approved the `confirm` calls being run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Approver {
    Human,
    Classifier,
}

impl Approver {
    fn as_str(self) -> &'static str {
        match self {
            Approver::Human => "human",
            Approver::Classifier => "classifier",
        }
    }
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
    /// The approval classifier, when a profile uses one.
    classifier: Option<Arc<Classifier>>,
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
            classifier: None,
        })
    }

    /// Use `classifier` for shadow and classifier approval. Without one,
    /// those modes ask a human for everything (as when it is down).
    pub fn with_classifier(mut self, classifier: Option<Arc<Classifier>>) -> Self {
        self.classifier = classifier;
        self
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
        let mut state = job.agent_state.clone().unwrap_or_default();
        let outcome = self.run_state(job, req, &mut state).await;
        // Every action the classifier approved is listed in what the user
        // is told at the end, however the job ends.
        let note = |text: String| with_auto_approved_note(text, &state);
        match outcome {
            Ok(AgentOutcome::Final(t)) => Ok(AgentOutcome::Final(note(t))),
            Ok(AgentOutcome::Refused(t)) => Ok(AgentOutcome::Refused(note(t))),
            Ok(AgentOutcome::GaveUp(t)) => Ok(AgentOutcome::GaveUp(note(t))),
            Err(AgentError::Permanent(e)) => Err(AgentError::Permanent(note(e))),
            other => other,
        }
    }

    async fn run_state(
        &self,
        job: &Job,
        req: &Request<'_>,
        state: &mut AgentState,
    ) -> Result<AgentOutcome, AgentError> {
        let (profile, filter) = self.profile(&job.profile)?;
        if let Some(text) = &state.final_text {
            // Finished before a restart; only the bookkeeping was lost.
            return Ok(AgentOutcome::Final(text.clone()));
        }

        if state.messages.is_empty() {
            self.start(job, req, profile, filter, state).await?;
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
            if pending.approval_mode == Some(ApprovalMode::Shadow) {
                self.compare_shadow(job, &pending, &decision).await;
            }
            let refusal_text = if decision == "expired" {
                "Not run: the user did not answer the confirmation request in time."
            } else {
                "Not run: the user declined this action."
            };
            let results = self
                .execute(
                    job,
                    state,
                    &pending.calls,
                    approved.then_some(Approver::Human),
                    refusal_text,
                )
                .await;
            state.messages.extend(llm.tool_results(&results));
            state.pending = None;
            self.db.save_agent_state(job.id, state).await?;
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
                    .handle_tool_turn(job, req, profile, filter, llm, state)
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
                        self.db.save_agent_state(job.id, state).await?;
                        continue;
                    }
                    return Err(AgentError::Permanent(e));
                }
            };
            self.audit(job, response_kind, turn.audit.clone()).await;

            match turn.stop {
                StopKind::Refusal(msg) => {
                    self.db.save_agent_state(job.id, state).await?;
                    return Ok(AgentOutcome::Refused(msg));
                }
                StopKind::MaxTokens => {
                    // A truncated tool call may still parse; never run it.
                    self.db.save_agent_state(job.id, state).await?;
                    return Err(AgentError::Permanent(format!(
                        "{}'s reply was cut off at max_tokens; raise max_tokens for {}",
                        state.provider.display(),
                        state.provider
                    )));
                }
                StopKind::Pause => {
                    state.messages.push(turn.message);
                    self.db.save_agent_state(job.id, state).await?;
                    continue;
                }
                StopKind::ToolUse => {
                    state.messages.push(turn.message);
                    self.db.save_agent_state(job.id, state).await?;
                    if let Some(outcome) = self
                        .handle_tool_turn(job, req, profile, filter, llm, state)
                        .await?
                    {
                        return Ok(outcome);
                    }
                }
                StopKind::EndTurn => {
                    let text = with_unavailable_note(llm.final_text(&turn.message), state, req);
                    state.messages.push(turn.message);
                    state.final_text = Some(text.clone());
                    self.db.save_agent_state(job.id, state).await?;
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
        profile: &ProfileConfig,
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

        // The policy of each call, kept for the approval step: `ask` calls
        // are never shown to the classifier.
        let mut policies = Vec::with_capacity(uses.len());
        let calls: Vec<PlannedCall> = uses
            .into_iter()
            .map(|u| {
                let name = u.name;
                let (input, bad_input) = match u.input {
                    Ok(v) => (v, None),
                    Err(why) => (Value::Null, Some(why)),
                };
                let mut policy = None;
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
                            let p = self.policies.policy_for(&q);
                            policy = Some(p);
                            match p {
                                Policy::Allow => CallAction::Run,
                                Policy::Confirm | Policy::Ask => CallAction::Confirm,
                                Policy::Deny => {
                                    CallAction::Reject("This tool is blocked by policy.".into())
                                }
                            }
                        }
                    }
                };
                policies.push(policy);
                PlannedCall {
                    tool_use_id: u.id,
                    claude_name: name,
                    input,
                    action,
                }
            })
            .collect();

        if calls.iter().any(|c| c.action == CallAction::Confirm) {
            let mode = self.config.approval_mode(Some(profile));
            let judgements = self
                .judge(req, llm, &last, state, &calls, &policies, mode)
                .await;
            // The batch runs without asking only if every call that needs
            // approval passed; one doubt sends the whole turn to the human.
            let auto = mode == ApprovalMode::Classifier
                && !judgements.is_empty()
                && judgements.iter().all(|j| j.outcome == Judged::Safe);
            let outcome = match (mode, auto) {
                (ApprovalMode::Classifier, true) => "auto_approved",
                (ApprovalMode::Classifier, false) => "escalated",
                (ApprovalMode::Shadow, _) => "shadow",
                (ApprovalMode::Manual, _) => "manual",
            };
            metrics::counter!("ringmaster_approvals_total", "mode" => mode.as_str(), "outcome" => outcome)
                .increment(1);
            self.audit_judgements(job, mode, outcome, &judgements).await;

            if auto {
                for j in &judgements {
                    let input = calls
                        .iter()
                        .find(|c| c.tool_use_id == j.tool_use_id)
                        .map(|c| c.input.clone())
                        .unwrap_or(Value::Null);
                    tracing::info!(job = job.id, tool = %j.tool, score = j.score.unwrap_or_default(), "auto-approved by the classifier");
                    state.auto_approved.push(AutoApproval {
                        tool: j.tool.clone(),
                        input,
                        score: j.score.unwrap_or_default(),
                    });
                }
                let results = self
                    .execute(job, state, &calls, Some(Approver::Classifier), "")
                    .await;
                state.messages.extend(llm.tool_results(&results));
                self.db.save_agent_state(job.id, state).await?;
                return Ok(None);
            }

            state.pending = Some(PendingTurn {
                calls: calls.clone(),
                approval_mode: (mode != ApprovalMode::Manual).then_some(mode),
                judgements: judgements.clone(),
            });
            self.db.save_agent_state(job.id, state).await?;
            self.request_confirmation(job, req, state, &calls, mode, &judgements)
                .await?;
            return Ok(Some(AgentOutcome::AwaitingConfirmation));
        }

        let results = self.execute(job, state, &calls, None, "").await;
        state.messages.extend(llm.tool_results(&results));
        self.db.save_agent_state(job.id, state).await?;
        Ok(None)
    }

    /// The classifier's say on each `confirm` call of a turn (nothing in
    /// manual mode). `ask` calls are never classified; in classifier mode
    /// one of them sends the turn to the human, so the others are not
    /// classified either. Calls are classified concurrently, each bounded
    /// by `approval.request_timeout_secs`. Never fails: no verdict is
    /// `Unavailable`, which asks the human.
    #[allow(clippy::too_many_arguments)]
    async fn judge(
        &self,
        req: &Request<'_>,
        llm: &dyn ChatModel,
        last: &Value,
        state: &AgentState,
        calls: &[PlannedCall],
        policies: &[Option<Policy>],
        mode: ApprovalMode,
    ) -> Vec<Judgement> {
        if mode == ApprovalMode::Manual {
            return Vec::new();
        }
        let pinned = calls
            .iter()
            .zip(policies)
            .any(|(c, p)| c.action == CallAction::Confirm && *p == Some(Policy::Ask));
        let skip_rest = pinned && mode == ApprovalMode::Classifier;
        let assistant_text = llm.reply_text(last);
        let futures = calls
            .iter()
            .zip(policies)
            .filter(|(c, _)| c.action == CallAction::Confirm)
            .map(|(c, p)| {
                let def = state.tools.iter().find(|d| d.claude_name == c.claude_name);
                let tool = def
                    .map(ToolDef::qualified)
                    .unwrap_or_else(|| c.claude_name.clone());
                let assistant_text = assistant_text.as_str();
                async move {
                    let judgement = |outcome, score, error| Judgement {
                        tool_use_id: c.tool_use_id.clone(),
                        tool: tool.clone(),
                        outcome,
                        score,
                        error,
                    };
                    if *p == Some(Policy::Ask) {
                        return judgement(Judged::Ask, None, None);
                    }
                    if skip_rest {
                        return judgement(Judged::Skipped, None, None);
                    }
                    let Some(classifier) = &self.classifier else {
                        return judgement(
                            Judged::Unavailable,
                            None,
                            Some(ClassifyError::NotConfigured.to_string()),
                        );
                    };
                    let call = ProposedCall {
                        transcript: req.transcript,
                        assistant_text: Some(assistant_text),
                        tool: &tool,
                        description: def.map(|d| d.description.as_str()).unwrap_or_default(),
                        arguments: &c.input,
                    };
                    match classifier.classify(&call).await {
                        Ok(score) => {
                            judgement(classifier.verdict(score).into(), Some(score), None)
                        }
                        Err(e) => {
                            tracing::warn!(tool = %tool, error = %e, "no classifier verdict; asking a human");
                            judgement(Judged::Unavailable, None, Some(e.to_string()))
                        }
                    }
                }
            });
        futures_util::future::join_all(futures).await
    }

    /// One `classification` audit row and the metrics per judged call.
    async fn audit_judgements(
        &self,
        job: &Job,
        mode: ApprovalMode,
        outcome: &str,
        judgements: &[Judgement],
    ) {
        let (threshold, cutoff) = match &self.classifier {
            Some(c) => (c.threshold(), c.cutoff()),
            None => (
                self.config.approval.threshold,
                self.config.approval.auto_approve_cutoff(),
            ),
        };
        for j in judgements {
            metrics::counter!("ringmaster_classifications_total", "mode" => mode.as_str(), "verdict" => j.outcome.as_str())
                .increment(1);
            if let Some(score) = j.score {
                metrics::histogram!("ringmaster_classifier_score", "mode" => mode.as_str())
                    .record(score);
            }
            self.audit(
                job,
                "classification",
                json!({
                    "tool": j.tool,
                    "mode": mode.as_str(),
                    "verdict": j.outcome.as_str(),
                    "score": j.score,
                    "threshold": threshold,
                    "auto_approve_below": cutoff,
                    "outcome": outcome,
                    "error": j.error,
                }),
            )
            .await;
        }
    }

    /// In shadow mode, record whether the classifier would have done what
    /// the human did. "would_auto_approve" means classifier mode would have
    /// run the turn without asking.
    async fn compare_shadow(&self, job: &Job, pending: &PendingTurn, decision: &str) {
        if decision != "approve" && decision != "deny" {
            return;
        }
        let would_auto = !pending.judgements.is_empty()
            && pending.judgements.iter().all(|j| j.outcome == Judged::Safe);
        let result = match (would_auto, decision == "approve") {
            (true, true) | (false, false) => "agree",
            // The one that matters: it would have run what the human refused.
            (true, false) => "classifier_would_have_run_a_denied_action",
            (false, true) => "classifier_would_have_asked",
        };
        metrics::counter!("ringmaster_shadow_comparisons_total", "result" => result).increment(1);
        if result != "agree" {
            tracing::info!(
                job = job.id,
                result,
                "shadow classifier disagreed with the human"
            );
        }
        self.audit(
            job,
            "shadow_comparison",
            json!({
                "decision": decision,
                "classifier_would_auto_approve": would_auto,
                "result": result,
                "disagreement": result != "agree",
                "judgements": pending.judgements,
            }),
        )
        .await;
    }

    async fn request_confirmation(
        &self,
        job: &Job,
        req: &Request<'_>,
        state: &AgentState,
        calls: &[PlannedCall],
        mode: ApprovalMode,
        judgements: &[Judgement],
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
                let mut line = format!("{q} {}", truncate(&c.input.to_string(), 300));
                if let Some(j) = judgements.iter().find(|j| j.tool_use_id == c.tool_use_id) {
                    line.push_str(&format!("\n  {}", self.judgement_note(mode, j)));
                }
                line
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
            json!({
                "summary": summary,
                "expires_at": expires_at,
                "approval_mode": mode.as_str(),
                "judgements": judgements,
            }),
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
        let flagged = mode == ApprovalMode::Classifier
            && judgements.iter().any(|j| j.outcome == Judged::Unsafe);
        let n = Notification {
            title: if flagged {
                "Ringmaster: approve? (flagged by the classifier)".into()
            } else {
                "Ringmaster: approve?".into()
            },
            body: format!("{summary}\n(expires in {} min)", ttl / 60),
            kind,
        };
        if let Err(e) = self.notifier.send(&n).await {
            tracing::error!(job = job.id, error = %e, "failed to send confirmation notification");
        }
        Ok(())
    }

    /// One line on what the classifier made of a call, for the
    /// confirmation message.
    fn judgement_note(&self, mode: ApprovalMode, j: &Judgement) -> String {
        let a = &self.config.approval;
        let (threshold, cutoff) = (a.threshold, a.auto_approve_cutoff());
        let score = j.score.unwrap_or_default();
        let shadow = if mode == ApprovalMode::Shadow {
            "(shadow) "
        } else {
            ""
        };
        match j.outcome {
            Judged::Safe if mode == ApprovalMode::Shadow => {
                format!("{shadow}Classifier would auto-approve this (score {score:.2})")
            }
            Judged::Safe => format!(
                "Classifier passed this (score {score:.2}); asking because of the other actions"
            ),
            Judged::Uncertain => format!(
                "{shadow}Classifier unsure (score {score:.2}; only below {cutoff:.2} runs without asking)"
            ),
            Judged::Unsafe => {
                format!("{shadow}Classifier flagged this (score {score:.2} > {threshold:.2})")
            }
            Judged::Unavailable => format!(
                "{shadow}No classifier verdict ({}); asking you instead",
                truncate(j.error.as_deref().unwrap_or("unknown error"), 120)
            ),
            Judged::Ask => "Always asks you (policy ask)".into(),
            Judged::Skipped => "Not checked: another action here always asks you".into(),
        }
    }

    /// Run planned calls in order; the results go back to the model in the
    /// provider's format.
    async fn execute(
        &self,
        job: &Job,
        state: &AgentState,
        calls: &[PlannedCall],
        approval: Option<Approver>,
        refusal_text: &str,
    ) -> Vec<ToolResult> {
        let approved = approval.is_some();
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
                    "approved_by": (approved && c.action == CallAction::Confirm)
                        .then(|| approval.map(Approver::as_str))
                        .flatten(),
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

/// When the classifier approved actions in this job, end the text with
/// them and their scores, so every action that ran without asking is seen.
fn with_auto_approved_note(text: String, state: &AgentState) -> String {
    if state.auto_approved.is_empty() {
        return text;
    }
    let items: Vec<String> = state
        .auto_approved
        .iter()
        .map(|a| {
            format!(
                "{} {} (score {:.2})",
                a.tool,
                truncate(&a.input.to_string(), 100),
                a.score
            )
        })
        .collect();
    format!(
        "{}\n\nAuto-approved by the classifier: {}",
        text.trim_end(),
        items.join("; ")
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
    fn states_without_approval_fields_still_load_and_stay_unchanged() {
        // A job parked before the classifier existed.
        let old = json!({
            "provider": "anthropic", "model": "m", "system": "s", "effort": "low", "tools": [],
            "messages": [], "iterations": 1, "final_text": null, "unavailable": [],
            "pending": {"calls": [{"tool_use_id": "t1", "claude_name": "home__x",
                                   "input": {}, "action": {"kind": "confirm"}}]}
        });
        let s: AgentState = serde_json::from_value(old.clone()).unwrap();
        assert!(s.auto_approved.is_empty());
        let p = s.pending.clone().unwrap();
        assert_eq!(p.approval_mode, None);
        assert!(p.judgements.is_empty());
        // Saved again in manual mode, it has the same shape as before.
        assert_eq!(serde_json::to_value(&s).unwrap(), old);
        // New fields round-trip.
        let mut s = s;
        s.auto_approved.push(AutoApproval {
            tool: "home/x".into(),
            input: json!({"a": 1}),
            score: 0.01,
        });
        s.pending = Some(PendingTurn {
            calls: p.calls,
            approval_mode: Some(ApprovalMode::Shadow),
            judgements: vec![Judgement {
                tool_use_id: "t1".into(),
                tool: "home/x".into(),
                outcome: Judged::Unsafe,
                score: Some(0.9),
                error: None,
            }],
        });
        let back: AgentState = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert_eq!(back.auto_approved, s.auto_approved);
        assert_eq!(back.pending.unwrap().judgements[0].outcome, Judged::Unsafe);
    }

    #[test]
    fn auto_approved_actions_are_listed_with_scores() {
        let mut s = AgentState::default();
        assert_eq!(with_auto_approved_note("Done.".into(), &s), "Done.");
        s.auto_approved.push(AutoApproval {
            tool: "home/set_light".into(),
            input: json!({"name": "kitchen"}),
            score: 0.031,
        });
        assert_eq!(
            with_auto_approved_note("Lights on. ".into(), &s),
            "Lights on.\n\nAuto-approved by the classifier: home/set_light {\"name\":\"kitchen\"} (score 0.03)"
        );
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
