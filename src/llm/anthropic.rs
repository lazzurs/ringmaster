//! Claude through the native Anthropic Messages API.
//!
//! The request shape is fixed by what Opus 5.5 expects: `output_config.effort`,
//! no `thinking` or sampling parameters, a cache breakpoint after the system
//! prompt plus automatic caching of the conversation tail, and server-side
//! refusal fallbacks where the model has them. Assistant `content` (thinking
//! blocks included) is stored and echoed back unchanged.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{ChatModel, ChatRequest, LlmError, Provider, StopKind, ToolCall, ToolResult, Turn};
use crate::claude::{ClaudeClient, FALLBACK_BETA};
use crate::config::{FallbackMode, ProviderSettings};
use crate::tools::ToolDef;

pub struct Anthropic {
    client: ClaudeClient,
    max_tokens: u32,
    fallbacks: FallbackMode,
    key_env: String,
}

impl Anthropic {
    pub fn new(s: &ProviderSettings, api_key: Option<String>) -> Result<Self> {
        Ok(Self {
            client: ClaudeClient::new(
                &s.base_url,
                api_key.unwrap_or_default(),
                s.request_timeout,
                s.max_retries,
            )?,
            max_tokens: s.max_tokens.unwrap_or(16000),
            fallbacks: s.refusal_fallbacks,
            key_env: s
                .api_key_env
                .clone()
                .unwrap_or_else(|| "ANTHROPIC_API_KEY".into()),
        })
    }

    pub fn body(&self, req: &ChatRequest<'_>) -> (Value, Vec<&'static str>) {
        let mut body = json!({
            "model": req.model,
            "max_tokens": self.max_tokens,
            "system": [{
                "type": "text",
                "text": req.system,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": req.messages,
            "output_config": {"effort": req.effort},
            // Automatic breakpoint for the growing conversation tail.
            "cache_control": {"type": "ephemeral"},
        });
        if req.effort.is_empty() {
            body.as_object_mut().map(|o| o.remove("output_config"));
        }
        if !req.tools.is_empty() {
            body["tools"] = Value::Array(req.tools.iter().map(ToolDef::to_api).collect());
        }
        let mut betas = Vec::new();
        if self.fallbacks.enabled_for(req.model) {
            body["fallbacks"] = json!("default");
            betas.push(FALLBACK_BETA);
        }
        (body, betas)
    }
}

#[async_trait]
impl ChatModel for Anthropic {
    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    fn user_message(&self, text: &str) -> Value {
        json!({"role": "user", "content": [{"type": "text", "text": text}]})
    }

    async fn complete(&self, req: &ChatRequest<'_>) -> Result<Turn, LlmError> {
        if !self.client.has_key() {
            return Err(LlmError::Permanent(format!(
                "no API key for anthropic: {} is not set",
                self.key_env
            )));
        }
        let (body, betas) = self.body(req);
        let resp = self.client.create_message(&body, &betas).await?;
        let stop_reason = resp
            .get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("");
        let content = sanitize_fallback(resp.get("content").cloned().unwrap_or(json!([])));
        let audit = summarize_response(&resp, &content);
        let stop = match stop_reason {
            "refusal" => {
                let category = resp
                    .pointer("/stop_details/category")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified");
                StopKind::Refusal(format!(
                    "Claude declined this request (category: {category})."
                ))
            }
            // A truncated tool_use input may still parse; never run it.
            "max_tokens" => StopKind::MaxTokens,
            "pause_turn" => StopKind::Pause,
            "tool_use" => StopKind::ToolUse,
            // end_turn, stop_sequence, or anything new: the answer.
            _ => StopKind::EndTurn,
        };
        Ok(Turn {
            stop,
            message: json!({"role": "assistant", "content": content}),
            audit,
        })
    }

    fn tool_calls(&self, message: &Value) -> Vec<ToolCall> {
        message
            .get("content")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                    .map(|u| ToolCall {
                        id: u
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: u
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        input: Ok(u.get("input").cloned().unwrap_or(json!({}))),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn final_text(&self, message: &Value) -> String {
        final_text(message.get("content").unwrap_or(&Value::Null))
    }

    /// All results of a turn go back in one user message.
    fn tool_results(&self, results: &[ToolResult]) -> Vec<Value> {
        let blocks: Vec<Value> = results
            .iter()
            .map(|r| {
                let mut block = json!({
                    "type": "tool_result",
                    "tool_use_id": r.id,
                    "content": r.text,
                });
                if r.is_error {
                    block["is_error"] = json!(true);
                }
                block
            })
            .collect();
        vec![json!({"role": "user", "content": blocks})]
    }

    fn audit_kinds(&self) -> (&'static str, &'static str) {
        ("claude_response", "claude_error")
    }
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
        "text": crate::agent::truncate(&final_text(content), 2000),
        "usage": resp.get("usage").map(|u| json!({
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        })),
    })
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
}
