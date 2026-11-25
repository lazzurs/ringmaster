//! Shared HTTP plumbing for every LLM provider: one JSON POST with retries.
//!
//! 429, 408, 5xx/529, connection errors and timeouts are transient and retried
//! with exponential back-off (honouring `retry-after`); other 4xx are
//! permanent. Quota and spend-cap errors are permanent even when they come
//! back as 429, since waiting does not fix them.

use std::time::{Duration, Instant};

use anyhow::Result;
use reqwest::StatusCode;
use serde_json::Value;

use super::{LlmError, Provider};

#[derive(Clone)]
pub struct Transport {
    http: reqwest::Client,
    provider: Provider,
    max_retries: u32,
    /// Upper bound for a single back-off sleep.
    max_backoff: Duration,
}

impl Transport {
    pub fn new(provider: Provider, timeout: Duration, max_retries: u32) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .user_agent(concat!("ringmaster/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            provider,
            max_retries,
            max_backoff: Duration::from_secs(60),
        })
    }

    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }

    /// POST `body` to `url` with `headers`, retrying transient failures.
    /// Returns the parsed JSON body of a 2xx response.
    pub async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> Result<Value, LlmError> {
        let mut attempt = 0u32;
        loop {
            let (retry_after, err) = match self.attempt(url, headers, body).await {
                Ok(v) => return Ok(v),
                Err(Failure::Permanent(e)) => return Err(LlmError::Permanent(e)),
                Err(Failure::Transient(retry_after, e)) => (retry_after, e),
            };
            if attempt >= self.max_retries {
                return Err(LlmError::Transient(err));
            }
            let backoff = retry_after.unwrap_or_else(|| {
                let base = Duration::from_millis(500) * 2u32.saturating_pow(attempt);
                base + Duration::from_millis(rand::random::<u64>() % 250)
            });
            let backoff = backoff.min(self.max_backoff);
            tracing::warn!(provider = self.provider.as_str(), attempt, ?backoff, error = %err, "retrying LLM request");
            tokio::time::sleep(backoff).await;
            attempt += 1;
        }
    }

    async fn attempt(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> Result<Value, Failure> {
        let provider = self.provider.as_str();
        let mut req = self
            .http
            .post(url)
            .header("content-type", "application/json")
            .json(body);
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let started = Instant::now();
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                metrics::counter!("ringmaster_llm_requests_total", "provider" => provider, "status" => "network_error")
                    .increment(1);
                let what = if e.is_timeout() {
                    "timed out"
                } else {
                    "request failed"
                };
                return Err(Failure::Transient(None, format!("{what}: {e}")));
            }
        };
        let status = resp.status();
        metrics::counter!("ringmaster_llm_requests_total", "provider" => provider, "status" => status.as_u16().to_string())
            .increment(1);
        metrics::histogram!("ringmaster_llm_request_seconds", "provider" => provider)
            .record(started.elapsed().as_secs_f64());
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s >= 0.0)
            .map(Duration::from_secs_f64);
        let request_id = ["request-id", "x-request-id"]
            .iter()
            .find_map(|h| resp.headers().get(*h))
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let text = resp
            .text()
            .await
            .map_err(|e| Failure::Transient(None, format!("reading response: {e}")))?;
        if status.is_success() {
            return serde_json::from_str::<Value>(&text)
                .map_err(|e| Failure::Transient(None, format!("invalid JSON in response: {e}")));
        }
        let detail = format!("{status}: {}", error_detail(&text, request_id.as_deref()));
        if is_quota_exhausted(&text) || !retryable(status) {
            Err(Failure::Permanent(detail))
        } else {
            Err(Failure::Transient(retry_after, detail))
        }
    }
}

enum Failure {
    Transient(Option<Duration>, String),
    Permanent(String),
}

fn retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
        || status.as_u16() == 529
}

/// Errors that look like rate limits but never clear by waiting.
fn is_quota_exhausted(body: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let s = |p: &str| v.pointer(p).and_then(Value::as_str);
    // Anthropic spend cap; OpenAI billing, spend and quota codes.
    s("/error/details/error_code") == Some("enforced_spend_limit_reached")
        || s("/error/type") == Some("insufficient_quota")
        || matches!(
            s("/error/code"),
            Some(
                "insufficient_quota"
                    | "credit_balance_exhausted"
                    | "organization_spend_limit_exceeded"
                    | "project_spend_limit_exceeded"
                    | "organization_usage_limit_exceeded"
            )
        )
}

/// A short, readable description of an error body from any provider.
pub fn error_detail(body: &str, request_id: Option<&str>) -> String {
    let msg = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            let s = |p: &str| v.pointer(p).and_then(Value::as_str).map(str::to_string);
            // Anthropic / OpenAI / OpenRouter: {"error": {"type"?, "message"}}
            if let Some(m) = s("/error/message") {
                return Some(match s("/error/type") {
                    Some(t) => format!("{t}: {m}"),
                    None => m,
                });
            }
            // Ollama: {"error": "..."}; Mistral: {"message": "..."} or {"detail": ...}
            s("/error")
                .or_else(|| s("/message"))
                .or_else(|| v.get("detail").map(|d| d.to_string()))
        })
        .unwrap_or_else(|| body.chars().take(300).collect());
    match request_id {
        Some(id) => format!("{msg} (request-id {id})"),
        None => msg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_detail_reads_every_provider_shape() {
        assert_eq!(
            error_detail(
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#,
                Some("req_1")
            ),
            "invalid_request_error: bad (request-id req_1)"
        );
        assert_eq!(
            error_detail(r#"{"error":{"code":402,"message":"no credits"}}"#, None),
            "no credits"
        );
        assert_eq!(
            error_detail(
                r#"{"error":"registry.ollama.ai/library/gemma does not support tools"}"#,
                None
            ),
            "registry.ollama.ai/library/gemma does not support tools"
        );
        assert_eq!(
            error_detail(r#"{"object":"error","message":"Unauthorized"}"#, None),
            "Unauthorized"
        );
        assert_eq!(error_detail("plain", None), "plain");
    }

    #[test]
    fn quota_errors_are_recognised() {
        assert!(is_quota_exhausted(
            r#"{"error":{"type":"rate_limit_error","details":{"error_code":"enforced_spend_limit_reached"}}}"#
        ));
        assert!(is_quota_exhausted(
            r#"{"error":{"message":"x","type":"insufficient_quota","code":"insufficient_quota"}}"#
        ));
        assert!(!is_quota_exhausted(
            r#"{"error":{"message":"slow down","type":"requests","code":"rate_limit_exceeded"}}"#
        ));
    }
}
