//! Minimal Claude Messages API client over raw HTTPS.
//!
//! Responses are kept as JSON values so assistant content (thinking blocks
//! included) can be echoed back byte-for-byte on the next turn.

use std::time::Duration;

use anyhow::Result;
use serde_json::Value;

use crate::llm::Provider;
use crate::llm::transport::Transport;

pub const API_VERSION: &str = "2026-10-21";
pub const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

/// Kept for callers of the Claude-only API; errors are shared by every
/// provider now.
pub use crate::llm::LlmError as ClaudeError;

#[derive(Clone)]
pub struct ClaudeClient {
    transport: Transport,
    base: String,
    api_key: String,
}

impl ClaudeClient {
    pub fn new(base: &str, api_key: String, timeout: Duration, max_retries: u32) -> Result<Self> {
        Ok(Self {
            transport: Transport::new(Provider::Anthropic, timeout, max_retries)?,
            base: base.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.transport = self.transport.with_max_backoff(d);
        self
    }

    pub fn has_key(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// POST /v1/messages with retries for 429, 5xx/529 and connection errors,
    /// honouring `retry-after`. 4xx (other than 408/429) are not retried.
    pub async fn create_message(&self, body: &Value, betas: &[&str]) -> Result<Value, ClaudeError> {
        let url = format!("{}/v1/messages", self.base);
        let mut headers = vec![
            ("x-api-key".to_string(), self.api_key.clone()),
            ("anthropic-version".to_string(), API_VERSION.to_string()),
        ];
        if !betas.is_empty() {
            headers.push(("anthropic-beta".to_string(), betas.join(",")));
        }
        let value = self.transport.post_json(&url, &headers, body).await?;
        record_usage(&value);
        Ok(value)
    }
}

fn record_usage(v: &Value) {
    let Some(u) = v.get("usage") else { return };
    for (field, kind) in [
        ("input_tokens", "input"),
        ("output_tokens", "output"),
        ("cache_read_input_tokens", "cache_read"),
        ("cache_creation_input_tokens", "cache_write"),
    ] {
        if let Some(n) = u.get(field).and_then(Value::as_u64) {
            metrics::counter!("ringmaster_llm_tokens_total", "provider" => "anthropic", "kind" => kind)
                .increment(n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn client(server: &MockServer, retries: u32) -> ClaudeClient {
        ClaudeClient::new(
            &server.uri(),
            "sk-test".into(),
            Duration::from_secs(5),
            retries,
        )
        .unwrap()
        .with_max_backoff(Duration::from_millis(20))
    }

    #[tokio::test]
    async fn retries_429_and_5xx_then_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(529).set_body_json(
                json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
            ))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("anthropic-version", API_VERSION))
            .and(header("x-api-key", "sk-test"))
            .and(header("anthropic-beta", FALLBACK_BETA))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"stop_reason":"end_turn","content":[]})),
            )
            .mount(&server)
            .await;
        let v = client(&server, 3)
            .create_message(&json!({}), &[FALLBACK_BETA])
            .await
            .unwrap();
        assert_eq!(v["stop_reason"], "end_turn");
    }

    #[tokio::test]
    async fn does_not_retry_400() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(
                json!({"type":"error","error":{"type":"invalid_request_error","message":"bad"}}),
            ))
            .expect(1)
            .mount(&server)
            .await;
        let err = client(&server, 3)
            .create_message(&json!({}), &[])
            .await
            .unwrap_err();
        assert!(
            matches!(err, ClaudeError::Permanent(ref m) if m.contains("invalid_request_error")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn gives_up_transiently_after_retries() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .expect(3)
            .mount(&server)
            .await;
        let err = client(&server, 2)
            .create_message(&json!({}), &[])
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::Transient(_)));
    }

    #[tokio::test]
    async fn spend_cap_is_permanent() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({"type":"error","error":{
                "type":"rate_limit_error","message":"cap","details":{"error_code":"enforced_spend_limit_reached"}}})))
            .expect(1)
            .mount(&server)
            .await;
        let err = client(&server, 3)
            .create_message(&json!({}), &[])
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::Permanent(_)));
    }
}
