//! LLM providers behind one interface.
//!
//! The agent loop is provider-neutral. Each adapter owns its native message
//! format: the conversation stored in a job is a list of that provider's own
//! messages, appended to and echoed back unchanged, so nothing is lost in
//! translation (Claude's thinking blocks, OpenRouter's `reasoning_details`).
//! An adapter turns a reply into a [`Turn`] (stop kind, tool calls, the
//! message to append) and writes tool results back in its own format.

pub mod anthropic;
pub mod chat;
pub mod transport;

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::tools::ToolDef;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Claude, through the native Anthropic Messages API.
    #[default]
    Anthropic,
    /// OpenAI Chat Completions.
    #[serde(rename = "openai")]
    OpenAi,
    /// Mistral AI chat completions.
    Mistral,
    /// OpenRouter (OpenAI-compatible, many upstream models).
    #[serde(rename = "openrouter")]
    OpenRouter,
    /// A local Ollama server (native `/api/chat`).
    Ollama,
}

impl Provider {
    pub const ALL: [Provider; 5] = [
        Provider::Anthropic,
        Provider::OpenAi,
        Provider::Mistral,
        Provider::OpenRouter,
        Provider::Ollama,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
            Provider::Mistral => "mistral",
            Provider::OpenRouter => "openrouter",
            Provider::Ollama => "ollama",
        }
    }

    /// How the provider is named in messages to the user.
    pub fn display(self) -> &'static str {
        match self {
            Provider::Anthropic => "Claude",
            Provider::OpenAi => "OpenAI",
            Provider::Mistral => "Mistral",
            Provider::OpenRouter => "OpenRouter",
            Provider::Ollama => "Ollama",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::OpenAi => "https://api.openai.com",
            Provider::Mistral => "https://api.mistral.ai",
            Provider::OpenRouter => "https://openrouter.ai/api",
            Provider::Ollama => "http://localhost:11434",
        }
    }

    /// Environment variable holding the API key, unless overridden with
    /// `api_key_env`. Ollama needs no key.
    pub fn default_key_env(self) -> Option<&'static str> {
        match self {
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::OpenAi => Some("OPENAI_API_KEY"),
            Provider::Mistral => Some("MISTRAL_API_KEY"),
            Provider::OpenRouter => Some("OPENROUTER_API_KEY"),
            Provider::Ollama => None,
        }
    }

    /// The only provider with a built-in default model.
    pub fn default_model(self) -> Option<&'static str> {
        match self {
            Provider::Anthropic => Some("claude-opus-5-5"),
            _ => None,
        }
    }

    /// Accepted `effort` values (checked by `check-config`).
    pub fn efforts(self) -> &'static [&'static str] {
        match self {
            Provider::Anthropic => &["low", "medium", "high", "xhigh", "max"],
            // `reasoning_effort`; support varies by model.
            Provider::OpenAi => &["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            // `reasoning_effort` on adjustable-reasoning models.
            Provider::Mistral => &["none", "high"],
            // `reasoning.effort`, normalised by OpenRouter across models.
            Provider::OpenRouter => &["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            // `think`: none turns thinking off.
            Provider::Ollama => &["none", "low", "medium", "high", "max"],
        }
    }

    pub fn needs_key(self) -> bool {
        self.default_key_env().is_some()
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Worth retrying later (rate limits, overload, outages, network).
    #[error("LLM API unavailable: {0}")]
    Transient(String),
    /// The request itself is wrong, or the account cannot make it.
    #[error("LLM API error: {0}")]
    Permanent(String),
}

/// What the agent asks for on one turn.
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub system: &'a str,
    /// Empty: provider default.
    pub effort: &'a str,
    pub tools: &'a [ToolDef],
    pub messages: &'a [Value],
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopKind {
    /// A final answer.
    EndTurn,
    /// The reply has tool calls to run.
    ToolUse,
    /// Cut off by the output limit: tool calls in it must never run.
    MaxTokens,
    /// Declined; the text is for the user.
    Refusal(String),
    /// Server-side pause (Anthropic `pause_turn`): send the history again.
    Pause,
}

/// A normalised reply.
#[derive(Debug, Clone)]
pub struct Turn {
    pub stop: StopKind,
    /// The assistant message to append to the conversation, in the
    /// provider's own format.
    pub message: Value,
    /// Summary for the audit log.
    pub audit: Value,
}

/// A tool call read from an assistant message.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// The arguments, or why they are unusable (malformed JSON, not an
    /// object). Unusable arguments are never run.
    pub input: Result<Value, String>,
}

/// The outcome of one tool call, to go back to the model.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub text: String,
    pub is_error: bool,
}

#[async_trait]
pub trait ChatModel: Send + Sync {
    fn provider(&self) -> Provider;

    /// The first user message (context and transcript).
    fn user_message(&self, text: &str) -> Value;

    /// One model call. Transient errors have already been retried.
    async fn complete(&self, req: &ChatRequest<'_>) -> Result<Turn, LlmError>;

    /// Tool calls in a stored assistant message.
    fn tool_calls(&self, message: &Value) -> Vec<ToolCall>;

    /// The text of a stored assistant message, trimmed; empty if it has
    /// none (a turn that only calls tools).
    fn reply_text(&self, message: &Value) -> String;

    /// The answer text of a stored assistant message ("Done." if empty).
    fn final_text(&self, message: &Value) -> String {
        let t = self.reply_text(message);
        if t.is_empty() { "Done.".into() } else { t }
    }

    /// Messages carrying the results of one turn's tool calls.
    fn tool_results(&self, results: &[ToolResult]) -> Vec<Value>;

    /// Audit log kinds for replies and errors.
    fn audit_kinds(&self) -> (&'static str, &'static str) {
        ("llm_response", "llm_error")
    }
}

/// One client per provider, built from the config at start-up. Jobs pick
/// theirs by the provider stored in their state.
#[derive(Clone)]
pub struct Llms {
    models: BTreeMap<Provider, Arc<dyn ChatModel>>,
}

impl Llms {
    /// `keys`: API keys by provider (missing keys only fail the requests
    /// that need them).
    pub fn new(config: &Config, keys: &BTreeMap<Provider, String>) -> anyhow::Result<Self> {
        let mut models: BTreeMap<Provider, Arc<dyn ChatModel>> = BTreeMap::new();
        for p in Provider::ALL {
            let settings = config.provider_settings(p);
            let key = keys.get(&p).cloned();
            let model: Arc<dyn ChatModel> = match p {
                Provider::Anthropic => Arc::new(anthropic::Anthropic::new(&settings, key)?),
                _ => Arc::new(chat::ChatCompletions::new(&settings, key)?),
            };
            models.insert(p, model);
        }
        Ok(Self { models })
    }

    /// Replace one provider's client (tests).
    pub fn with(mut self, model: Arc<dyn ChatModel>) -> Self {
        self.models.insert(model.provider(), model);
        self
    }

    pub fn get(&self, p: Provider) -> &dyn ChatModel {
        self.models
            .get(&p)
            .map(|m| m.as_ref())
            .expect("a client for every provider")
    }
}

/// Parse tool-call arguments given as a JSON string (or, from Ollama, an
/// object). Anything but a JSON object is unusable.
pub fn parse_arguments(raw: Option<&Value>) -> Result<Value, String> {
    let v = match raw {
        None | Some(Value::Null) => return Ok(Value::Object(Default::default())),
        Some(Value::String(s)) if s.trim().is_empty() => {
            return Ok(Value::Object(Default::default()));
        }
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).map_err(|e| {
            format!(
                "the arguments are not valid JSON ({e}): {}",
                crate::agent::truncate(s, 200)
            )
        })?,
        Some(v) => v.clone(),
    };
    match v {
        Value::Object(_) => Ok(v),
        other => Err(format!(
            "the arguments must be a JSON object, not {}",
            match other {
                Value::Array(_) => "an array",
                Value::String(_) => "a string",
                Value::Number(_) => "a number",
                Value::Bool(_) => "a boolean",
                _ => "null",
            }
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_names_round_trip() {
        for p in Provider::ALL {
            let v = serde_json::to_value(p).unwrap();
            assert_eq!(v, json!(p.as_str()));
            assert_eq!(serde_json::from_value::<Provider>(v).unwrap(), p);
        }
        assert!(serde_json::from_value::<Provider>(json!("gemini")).is_err());
    }

    #[test]
    fn arguments_must_be_a_json_object() {
        assert_eq!(
            parse_arguments(Some(&json!("{\"a\":1}"))),
            Ok(json!({"a": 1}))
        );
        assert_eq!(parse_arguments(Some(&json!({"a": 1}))), Ok(json!({"a": 1})));
        assert_eq!(parse_arguments(Some(&json!(""))), Ok(json!({})));
        assert_eq!(parse_arguments(None), Ok(json!({})));
        assert!(parse_arguments(Some(&json!("{\"a\":"))).is_err());
        assert!(parse_arguments(Some(&json!("[1]"))).is_err());
        assert!(parse_arguments(Some(&json!("\"x\""))).is_err());
    }
}
