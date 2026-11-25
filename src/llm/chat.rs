//! OpenAI-style chat with function calling, for OpenAI, Mistral, OpenRouter
//! (`/v1/chat/completions`) and Ollama (native `/api/chat`, which shares the
//! message shapes but can set the context size per request).
//!
//! Messages are stored as each provider sent them, with a whitelist of fields
//! (so response-only fields are not echoed back) and tool-call ids filled in
//! when a model left them out.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};

use super::transport::Transport;
use super::{ChatModel, ChatRequest, LlmError, Provider, StopKind, ToolCall, ToolResult, Turn};
use crate::config::{ProviderSettings, credential_header};
use crate::tools::ToolDef;

/// How one provider differs from the common shape.
#[derive(Debug, Clone, Copy)]
struct Dialect {
    path: &'static str,
    /// Role of the system prompt: OpenAI wants `developer` for current models.
    system_role: &'static str,
    /// Field for the output limit; Ollama uses `options.num_predict`.
    max_tokens_field: &'static str,
    /// Ollama's native API: top-level `message`/`done_reason`, objects for
    /// tool-call arguments, `options`, `think`.
    ollama_native: bool,
    /// Name the tool on a tool result message (Mistral: `name`, Ollama:
    /// `tool_name`).
    tool_result_name: Option<&'static str>,
    /// Extra assistant-message fields that must be echoed back unchanged.
    echo: &'static [&'static str],
}

fn dialect(p: Provider) -> Dialect {
    let common = Dialect {
        path: "/v1/chat/completions",
        system_role: "system",
        max_tokens_field: "max_tokens",
        ollama_native: false,
        tool_result_name: None,
        echo: &[],
    };
    match p {
        Provider::OpenAi => Dialect {
            system_role: "developer",
            // `max_tokens` is deprecated and rejected by reasoning models.
            max_tokens_field: "max_completion_tokens",
            ..common
        },
        Provider::Mistral => Dialect {
            tool_result_name: Some("name"),
            ..common
        },
        Provider::OpenRouter => Dialect {
            // Reasoning must be passed back unchanged across tool calls.
            echo: &["reasoning_details"],
            ..common
        },
        Provider::Ollama => Dialect {
            path: "/api/chat",
            max_tokens_field: "num_predict",
            ollama_native: true,
            tool_result_name: Some("tool_name"),
            echo: &["thinking"],
            ..common
        },
        Provider::Anthropic => common,
    }
}

/// The key is held only as a sensitive header value, so `Debug` shows
/// `Sensitive` instead of it.
#[derive(Debug)]
pub struct ChatCompletions {
    provider: Provider,
    dialect: Dialect,
    transport: Transport,
    url: String,
    headers: HeaderMap,
    key_env: Option<String>,
    missing_key: bool,
    /// A key was set but cannot be sent as a header.
    invalid_key: bool,
    max_tokens: Option<u32>,
    num_ctx: Option<u32>,
}

impl ChatCompletions {
    pub fn new(s: &ProviderSettings, api_key: Option<String>) -> Result<Self> {
        let d = dialect(s.provider);
        let base = s.base_url.trim_end_matches('/');
        // Accept a base URL given with the API version on the end.
        let base = base.strip_suffix("/v1").unwrap_or(base);
        let mut headers = HeaderMap::new();
        let mut invalid_key = false;
        // Every provider, Ollama behind a reverse proxy included, gets the
        // key as a bearer token.
        if let Some(k) = &api_key {
            match credential_header(&format!("Bearer {k}")) {
                Some(v) => {
                    headers.insert(AUTHORIZATION, v);
                }
                None => invalid_key = true,
            }
        }
        // OpenRouter attribution: public values from the config, not
        // secrets (checked to be valid header values by `Config::validate`).
        for (name, value) in [
            ("http-referer", &s.app_url),
            ("x-openrouter-title", &s.app_title),
        ] {
            if let Some(v) = value.as_deref().and_then(|v| HeaderValue::from_str(v).ok()) {
                headers.insert(HeaderName::from_static(name), v);
            }
        }
        Ok(Self {
            provider: s.provider,
            dialect: d,
            transport: Transport::new(s.provider, s.request_timeout, s.max_retries)?,
            url: format!("{base}{}", d.path),
            headers,
            key_env: s.api_key_env.clone(),
            missing_key: api_key.is_none() && s.provider.needs_key(),
            invalid_key,
            max_tokens: s.max_tokens,
            num_ctx: s.num_ctx,
        })
    }

    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.transport = self.transport.with_max_backoff(d);
        self
    }

    fn body(&self, req: &ChatRequest<'_>) -> Value {
        let d = &self.dialect;
        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        messages.push(json!({"role": d.system_role, "content": req.system}));
        messages.extend(req.messages.iter().cloned());
        let mut body = json!({"model": req.model, "messages": messages});
        if !req.tools.is_empty() {
            body["tools"] = Value::Array(req.tools.iter().map(function_tool).collect());
        }
        if d.ollama_native {
            body["stream"] = json!(false);
            // Fail loudly instead of silently dropping the start of the
            // conversation when it outgrows the context window.
            body["truncate"] = json!(false);
            body["shift"] = json!(false);
            let mut options = Map::new();
            if let Some(n) = self.num_ctx {
                options.insert("num_ctx".into(), json!(n));
            }
            if let Some(n) = self.max_tokens {
                options.insert(d.max_tokens_field.into(), json!(n));
            }
            if !options.is_empty() {
                body["options"] = Value::Object(options);
            }
        } else if let Some(n) = self.max_tokens {
            body[d.max_tokens_field] = json!(n);
        }
        if !req.effort.is_empty() {
            match self.provider {
                Provider::OpenRouter => body["reasoning"] = json!({"effort": req.effort}),
                Provider::Ollama => {
                    body["think"] = if req.effort == "none" {
                        json!(false)
                    } else {
                        json!(req.effort)
                    }
                }
                _ => body["reasoning_effort"] = json!(req.effort),
            }
        }
        body
    }

    /// Normalise a successful response body.
    fn turn(&self, resp: &Value, first_turn: bool) -> Result<Turn, LlmError> {
        let name = self.provider.display();
        if let Some(err) = resp.get("error").filter(|e| !e.is_null()) {
            // OpenRouter can report a failed generation inside a 200.
            let code = err.get("code").and_then(Value::as_u64).unwrap_or(0);
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| err.to_string());
            let msg =
                super::transport::redact(&format!("{name} error {code}: {msg}"), &self.headers);
            return Err(
                if (400..500).contains(&code) && code != 408 && code != 429 {
                    LlmError::Permanent(msg)
                } else {
                    LlmError::Transient(msg)
                },
            );
        }
        let (message, finish) = if self.dialect.ollama_native {
            (resp.get("message"), resp.get("done_reason"))
        } else {
            let choice = resp.pointer("/choices/0");
            (
                choice.and_then(|c| c.get("message")),
                choice.and_then(|c| c.get("finish_reason")),
            )
        };
        let Some(message) = message.and_then(Value::as_object) else {
            return Err(LlmError::Transient(format!(
                "{name} response has no message"
            )));
        };
        let finish = finish.and_then(Value::as_str).unwrap_or("").to_string();
        record_usage(self.provider, resp);

        let stored = self.store(message);
        let calls = self.tool_calls(&stored);
        let text = content_text(stored.get("content"));
        let refusal = message
            .get("refusal")
            .and_then(Value::as_str)
            .filter(|r| !r.trim().is_empty());

        let stop = if let Some(r) = refusal {
            StopKind::Refusal(format!("{name} declined this request: {}", r.trim()))
        } else {
            match finish.as_str() {
                "content_filter" => {
                    StopKind::Refusal(format!("{name} declined this request (content filter)."))
                }
                "length" | "model_length" => StopKind::MaxTokens,
                "error" => {
                    return Err(LlmError::Transient(format!(
                        "{name} reported an error while generating"
                    )));
                }
                "load" | "unload" => {
                    return Err(LlmError::Transient(format!(
                        "{name} was loading the model and gave no answer"
                    )));
                }
                _ if !calls.is_empty() => StopKind::ToolUse,
                _ if text.is_empty() && first_turn => {
                    // A small model whose tool call could not be parsed
                    // comes back empty; "Done." would claim success.
                    return Err(LlmError::Permanent(format!(
                        "{name} returned an empty reply (no text and no tool calls); \
                         a small local model may have produced a malformed tool call"
                    )));
                }
                _ => StopKind::EndTurn,
            }
        };

        let audit = json!({
            "provider": self.provider,
            "id": resp.get("id"),
            "model": resp.get("model"),
            "finish_reason": finish,
            "native_finish_reason": resp.pointer("/choices/0/native_finish_reason"),
            "tool_uses": calls.iter().map(|c| json!({
                "name": c.name,
                "input": match &c.input { Ok(v) => v.clone(), Err(e) => json!({"invalid": e}) },
            })).collect::<Vec<_>>(),
            "refusal": refusal,
            "text": crate::agent::truncate(&text, 2000),
            "usage": usage_summary(resp),
        });
        Ok(Turn {
            stop,
            message: stored,
            audit,
        })
    }

    /// The assistant message as it will be echoed back.
    fn store(&self, m: &Map<String, Value>) -> Value {
        let mut out = Map::new();
        out.insert("role".into(), json!("assistant"));
        let calls: Vec<Value> = m
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(with_id).collect())
            .unwrap_or_default();
        let content = match m.get("content") {
            None | Some(Value::Null) => json!(""),
            Some(c) => c.clone(),
        };
        out.insert("content".into(), content);
        if !calls.is_empty() {
            out.insert("tool_calls".into(), Value::Array(calls));
        }
        for k in self.dialect.echo {
            if let Some(v) = m.get(*k).filter(|v| !v.is_null()) {
                out.insert((*k).into(), v.clone());
            }
        }
        Value::Object(out)
    }
}

#[async_trait]
impl ChatModel for ChatCompletions {
    fn provider(&self) -> Provider {
        self.provider
    }

    fn user_message(&self, text: &str) -> Value {
        json!({"role": "user", "content": text})
    }

    async fn complete(&self, req: &ChatRequest<'_>) -> Result<Turn, LlmError> {
        if self.missing_key {
            return Err(LlmError::Permanent(format!(
                "no API key for {}: {} is not set",
                self.provider,
                self.key_env.as_deref().unwrap_or("the key variable")
            )));
        }
        if self.invalid_key {
            return Err(LlmError::Permanent(format!(
                "the API key for {} in {} is not a valid HTTP header value (check for stray characters)",
                self.provider,
                self.key_env.as_deref().unwrap_or("the key variable")
            )));
        }
        let body = self.body(req);
        let resp = self
            .transport
            .post_json(&self.url, &self.headers, &body)
            .await
            .map_err(|e| match e {
                // Ollama answers 500 when the model's tool call is not valid
                // JSON. Retrying the job would just repeat it: fail cleanly.
                LlmError::Transient(m)
                    if self.dialect.ollama_native && m.contains("failed to parse") =>
                {
                    LlmError::Permanent(format!(
                        "the model produced a malformed tool call ({m}); nothing was run"
                    ))
                }
                other => other,
            })?;
        self.turn(&resp, req.messages.len() == 1)
    }

    fn tool_calls(&self, message: &Value) -> Vec<ToolCall> {
        message
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|c| ToolCall {
                        id: c
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: c
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        input: super::parse_arguments(c.pointer("/function/arguments")),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn reply_text(&self, message: &Value) -> String {
        content_text(message.get("content")).trim().to_string()
    }

    fn tool_results(&self, results: &[ToolResult]) -> Vec<Value> {
        results
            .iter()
            .map(|r| {
                let content = if r.is_error {
                    format!("Error: {}", r.text)
                } else {
                    r.text.clone()
                };
                let mut m = json!({"role": "tool", "tool_call_id": r.id, "content": content});
                if let Some(field) = self.dialect.tool_result_name {
                    m[field] = json!(r.name);
                }
                m
            })
            .collect()
    }
}

fn function_tool(t: &ToolDef) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.claude_name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

/// A tool call with an id. Models that leave it out get a generated one
/// (nine alphanumerics: the strictest format Mistral accepts).
fn with_id(call: &Value) -> Value {
    let mut c = call.clone();
    let has_id = c
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if !has_id && let Some(o) = c.as_object_mut() {
        const ALNUM: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        let id: String = (0..9)
            .map(|_| ALNUM[rand::random::<u32>() as usize % ALNUM.len()] as char)
            .collect();
        o.insert("id".into(), json!(id));
    }
    if c.get("type").is_none()
        && let Some(o) = c.as_object_mut()
    {
        o.insert("type".into(), json!("function"));
    }
    c
}

/// Text of a message's `content`: a string, or (Mistral reasoning models)
/// an array of chunks of which only `text` ones are the answer.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Array(chunks)) => chunks
            .iter()
            .filter(|c| c.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string(),
        _ => String::new(),
    }
}

fn usage_numbers(resp: &Value) -> [(&'static str, Option<u64>); 3] {
    let n = |p: &str| resp.pointer(p).and_then(Value::as_u64);
    [
        (
            "input",
            n("/usage/prompt_tokens").or_else(|| n("/prompt_eval_count")),
        ),
        (
            "output",
            n("/usage/completion_tokens").or_else(|| n("/eval_count")),
        ),
        (
            "cache_read",
            n("/usage/prompt_tokens_details/cached_tokens"),
        ),
    ]
}

fn record_usage(p: Provider, resp: &Value) {
    for (kind, n) in usage_numbers(resp) {
        if let Some(n) = n {
            metrics::counter!("ringmaster_llm_tokens_total", "provider" => p.as_str(), "kind" => kind)
                .increment(n);
        }
    }
}

fn usage_summary(resp: &Value) -> Value {
    let [(_, input), (_, output), (_, cached)] = usage_numbers(resp);
    json!({"input_tokens": input, "output_tokens": output, "cached_tokens": cached})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn adapter(p: Provider) -> ChatCompletions {
        let cfg = Config::from_toml("").unwrap();
        ChatCompletions::new(&cfg.provider_settings(p), Some("k".into())).unwrap()
    }

    fn reply(message: Value, finish: &str) -> Value {
        json!({"id": "x", "model": "m", "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
    }

    #[test]
    fn tool_call_turn_is_normalised_and_echoable() {
        let a = adapter(Provider::OpenAi);
        let t = a
            .turn(
                &reply(
                    json!({"role": "assistant", "content": null, "refusal": null, "annotations": [],
                       "tool_calls": [{"id": "call_1", "type": "function",
                                       "function": {"name": "home__set_light", "arguments": "{\"name\":\"kitchen\",\"on\":true}"}}]}),
                    "tool_calls",
                ),
                true,
            )
            .unwrap();
        assert_eq!(t.stop, StopKind::ToolUse);
        // Response-only fields are not echoed; null content becomes "".
        assert_eq!(
            t.message,
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "home__set_light", "arguments": "{\"name\":\"kitchen\",\"on\":true}"}}]})
        );
        let calls = a.tool_calls(&t.message);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].input, Ok(json!({"name": "kitchen", "on": true})));
    }

    #[test]
    fn stop_kinds() {
        let a = adapter(Provider::OpenAi);
        let msg = json!({"role": "assistant", "content": "hi"});
        assert_eq!(
            a.turn(&reply(msg.clone(), "stop"), true).unwrap().stop,
            StopKind::EndTurn
        );
        assert_eq!(
            a.turn(&reply(msg.clone(), "length"), true).unwrap().stop,
            StopKind::MaxTokens
        );
        assert_eq!(
            adapter(Provider::Mistral)
                .turn(&reply(msg.clone(), "model_length"), true)
                .unwrap()
                .stop,
            StopKind::MaxTokens
        );
        assert!(matches!(
            a.turn(&reply(msg.clone(), "content_filter"), true)
                .unwrap()
                .stop,
            StopKind::Refusal(_)
        ));
        let refused =
            json!({"role": "assistant", "content": null, "refusal": "I can't help with that."});
        assert_eq!(
            a.turn(&reply(refused, "stop"), true).unwrap().stop,
            StopKind::Refusal("OpenAI declined this request: I can't help with that.".into())
        );
        assert!(matches!(
            a.turn(&reply(msg, "error"), true),
            Err(LlmError::Transient(_))
        ));
        // Empty first reply: an error, not "Done.".
        let empty = json!({"role": "assistant", "content": ""});
        assert!(matches!(
            a.turn(&reply(empty.clone(), "stop"), true),
            Err(LlmError::Permanent(_))
        ));
        assert_eq!(
            a.turn(&reply(empty, "stop"), false).unwrap().stop,
            StopKind::EndTurn
        );
        // OpenRouter error inside a 200.
        let or = adapter(Provider::OpenRouter);
        assert!(matches!(
            or.turn(
                &json!({"error": {"code": 502, "message": "upstream down"}}),
                true
            ),
            Err(LlmError::Transient(_))
        ));
        assert!(matches!(
            or.turn(
                &json!({"error": {"code": 403, "message": "moderation"}}),
                true
            ),
            Err(LlmError::Permanent(_))
        ));
    }

    #[test]
    fn malformed_arguments_are_flagged_not_run() {
        let a = adapter(Provider::Ollama);
        let t = a
            .turn(
                &json!({"model": "qwen3:8b", "done": true, "done_reason": "stop",
                        "message": {"role": "assistant", "content": "",
                                    "tool_calls": [{"function": {"name": "x", "arguments": "{\"a\": "}}]}}),
                true,
            )
            .unwrap();
        assert_eq!(t.stop, StopKind::ToolUse);
        let calls = a.tool_calls(&t.message);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].input.is_err());
        // An id was generated and stored so the result can refer to it.
        assert_eq!(calls[0].id.len(), 9);
        assert_eq!(t.message["tool_calls"][0]["id"], calls[0].id.as_str());
    }

    #[test]
    fn mistral_thinking_chunks_are_kept_and_skipped_for_text() {
        let a = adapter(Provider::Mistral);
        let content = json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]},
            {"type": "text", "text": "Lights on."}
        ]);
        let t = a
            .turn(
                &reply(
                    json!({"role": "assistant", "content": content.clone()}),
                    "stop",
                ),
                true,
            )
            .unwrap();
        assert_eq!(t.message["content"], content);
        assert_eq!(a.final_text(&t.message), "Lights on.");
    }

    #[test]
    fn tool_results_per_dialect() {
        let r = [ToolResult {
            id: "abc123XYZ".into(),
            name: "home__set_light".into(),
            text: "declined".into(),
            is_error: true,
        }];
        assert_eq!(
            adapter(Provider::OpenAi).tool_results(&r)[0],
            json!({"role": "tool", "tool_call_id": "abc123XYZ", "content": "Error: declined"})
        );
        assert_eq!(
            adapter(Provider::Mistral).tool_results(&r)[0]["name"],
            "home__set_light"
        );
        assert_eq!(
            adapter(Provider::Ollama).tool_results(&r)[0]["tool_name"],
            "home__set_light"
        );
    }

    #[test]
    fn base_url_with_v1_is_accepted() {
        let mut cfg = Config::from_toml("").unwrap();
        cfg.llm.ollama.base_url = Some("http://mac.lan:11434/".into());
        let a = ChatCompletions::new(&cfg.provider_settings(Provider::Ollama), None).unwrap();
        assert_eq!(a.url, "http://mac.lan:11434/api/chat");
        cfg.llm.openrouter.base_url = Some("https://openrouter.ai/api/v1".into());
        let a = ChatCompletions::new(&cfg.provider_settings(Provider::OpenRouter), None).unwrap();
        assert_eq!(a.url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn every_provider_sends_its_key_as_a_sensitive_header() {
        const KEY: &str = "chat-key-0123456789abcdef";
        let mut cfg = Config::from_toml("").unwrap();
        cfg.llm.openrouter.app_url = Some("https://github.com/lazzurs/ringmaster".into());
        cfg.llm.openrouter.app_title = Some("Ringmaster".into());
        // Ollama sends a key when it sits behind a reverse proxy.
        for p in [
            Provider::OpenAi,
            Provider::Mistral,
            Provider::OpenRouter,
            Provider::Ollama,
        ] {
            let a = ChatCompletions::new(&cfg.provider_settings(p), Some(KEY.into())).unwrap();
            let req = a.transport.request(&a.url, &a.headers, &json!({})).unwrap();
            let auth = &req.headers()[AUTHORIZATION];
            assert_eq!(auth, format!("Bearer {KEY}").as_str(), "{p}");
            assert!(auth.is_sensitive(), "{p}");
            let shown = format!("{a:?} {req:?} {:?}", req.headers());
            assert!(!shown.contains(KEY), "{p}: {shown}");
            if p == Provider::OpenRouter {
                assert_eq!(req.headers()["x-openrouter-title"], "Ringmaster");
                assert_eq!(
                    req.headers()["http-referer"],
                    "https://github.com/lazzurs/ringmaster"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_key_that_is_not_a_header_value_fails_only_its_requests() {
        let cfg = Config::from_toml("").unwrap();
        let a = ChatCompletions::new(
            &cfg.provider_settings(Provider::OpenAi),
            Some("sk-bad\u{7f}key".into()),
        )
        .unwrap();
        let err = a
            .complete(&ChatRequest {
                model: "m",
                system: "s",
                effort: "",
                tools: &[],
                messages: &[json!({"role": "user", "content": "hi"})],
            })
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, LlmError::Permanent(_)), "{msg}");
        assert!(
            msg.contains("OPENAI_API_KEY") && !msg.contains("sk-bad"),
            "{msg}"
        );
    }

    #[test]
    fn errors_inside_a_200_are_redacted() {
        const KEY: &str = "sk-or-0123456789abcdef";
        let cfg = Config::from_toml("").unwrap();
        let a = ChatCompletions::new(
            &cfg.provider_settings(Provider::OpenRouter),
            Some(KEY.into()),
        )
        .unwrap();
        let err = a
            .turn(
                &json!({"error": {"code": 401, "message": format!("bad key {KEY}")}}),
                true,
            )
            .unwrap_err();
        assert!(!err.to_string().contains(KEY), "{err}");
    }
}
