//! Minimal Claude Messages API client over raw HTTPS.
//!
//! Responses are kept as JSON values so assistant content (thinking blocks
//! included) can be echoed back byte-for-byte on the next turn.

use std::time::Duration;

use anyhow::Result;
use reqwest::StatusCode;
use serde_json::Value;

pub const API_VERSION: &str = "2026-10-21";
pub const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

#[derive(Debug, thiserror::Error)]
pub enum ClaudeError {
    /// Worth retrying later (rate limits, overload, outages, network).
    #[error("Claude API unavailable: {0}")]
    Transient(String),
    /// The request itself is wrong, or the account cannot make it.
    #[error("Claude API error: {0}")]
    Permanent(String),
}

#[derive(Clone)]
pub struct ClaudeClient {
    http: reqwest::Client,
    base: String,
    api_key: String,
    max_retries: u32,
    /// Upper bound for a single back-off sleep.
    max_backoff: Duration,
}

impl ClaudeClient {
    pub fn new(base: &str, api_key: String, timeout: Duration, max_retries: u32) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .user_agent(concat!("ringmaster/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_string(),
            api_key,
            max_retries,
            max_backoff: Duration::from_secs(60),
        })
    }

    #[cfg(test)]
    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }

    /// POST /v1/messages with retries for 429, 5xx/529 and connection errors,
    /// honouring `retry-after`. 4xx (other than 408/429) are not retried.
    pub async fn create_message(&self, body: &Value, betas: &[&str]) -> Result<Value, ClaudeError> {
        let url = format!("{}/v1/messages", self.base);
        let mut attempt = 0u32;
        loop {
            let mut req = self
                .http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .json(body);
            if !betas.is_empty() {
                req = req.header("anthropic-beta", betas.join(","));
            }
            let started = std::time::Instant::now();
            let outcome = req.send().await;
            let (retry_after, err) = match outcome {
                Err(e) => {
                    metrics::counter!("ringmaster_claude_requests_total", "status" => "network_error")
                        .increment(1);
                    (None, ClaudeError::Transient(format!("request failed: {e}")))
                }
                Ok(resp) => {
                    let status = resp.status();
                    metrics::counter!("ringmaster_claude_requests_total", "status" => status.as_u16().to_string())
                        .increment(1);
                    metrics::histogram!("ringmaster_claude_request_seconds")
                        .record(started.elapsed().as_secs_f64());
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.trim().parse::<f64>().ok())
                        .map(Duration::from_secs_f64);
                    let request_id = resp
                        .headers()
                        .get("request-id")
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string);
                    let text = resp
                        .text()
                        .await
                        .map_err(|e| ClaudeError::Transient(format!("reading response: {e}")))?;
                    if status.is_success() {
                        let value: Value = serde_json::from_str(&text).map_err(|e| {
                            ClaudeError::Transient(format!("invalid JSON in response: {e}"))
                        })?;
                        record_usage(&value);
                        return Ok(value);
                    }
                    let detail = error_detail(&text, request_id.as_deref());
                    if is_spend_cap(&text) {
                        return Err(ClaudeError::Permanent(format!("{status}: {detail}")));
                    }
                    if retryable(status) {
                        (
                            retry_after,
                            ClaudeError::Transient(format!("{status}: {detail}")),
                        )
                    } else {
                        return Err(ClaudeError::Permanent(format!("{status}: {detail}")));
                    }
                }
            };
            if attempt >= self.max_retries {
                return Err(err);
            }
            let backoff = retry_after.unwrap_or_else(|| {
                let base = Duration::from_millis(500) * 2u32.saturating_pow(attempt);
                base + Duration::from_millis(rand::random::<u64>() % 250)
            });
            let backoff = backoff.min(self.max_backoff);
            tracing::warn!(attempt, ?backoff, error = %err, "retrying Claude request");
            tokio::time::sleep(backoff).await;
            attempt += 1;
        }
    }
}

fn retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
        || status.as_u16() == 529
}

fn is_spend_cap(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/details/error_code")
                .and_then(Value::as_str)
                .map(|c| c == "enforced_spend_limit_reached")
        })
        .unwrap_or(false)
}

fn error_detail(body: &str, request_id: Option<&str>) -> String {
    let msg = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            let t = v
                .pointer("/error/type")
                .and_then(Value::as_str)?
                .to_string();
            let m = v
                .pointer("/error/message")
                .and_then(Value::as_str)?
                .to_string();
            Some(format!("{t}: {m}"))
        })
        .unwrap_or_else(|| body.chars().take(300).collect());
    match request_id {
        Some(id) => format!("{msg} (request-id {id})"),
        None => msg,
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
            metrics::counter!("ringmaster_claude_tokens_total", "kind" => kind).increment(n);
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
