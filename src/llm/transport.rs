//! Shared HTTP plumbing for every LLM provider: one JSON POST with retries.
//!
//! 429, 408, 5xx/529, connection errors and timeouts are transient and retried
//! with exponential back-off (honouring `retry-after`); other 4xx are
//! permanent. Quota and spend-cap errors are permanent even when they come
//! back as 429, since waiting does not fix them.

use std::time::{Duration, Instant};

use anyhow::Result;
use reqwest::StatusCode;
use reqwest::header::{CONTENT_TYPE, HeaderMap, LOCATION};
use serde_json::Value;

use super::{LlmError, Provider};

/// Credentials travel only in header values marked sensitive (see
/// `config::credential_header`), so `Debug` output of this, of the clients
/// holding it and of the requests it builds never shows them.
#[derive(Debug, Clone)]
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
            // API endpoints do not redirect. Following one would replay the
            // key headers (`x-api-key` survives even a cross-host redirect)
            // to wherever it points.
            .redirect(reqwest::redirect::Policy::none())
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
    /// Returns the parsed JSON body of a 2xx response. Error messages have
    /// the sensitive header values in `headers` redacted.
    pub async fn post_json(
        &self,
        url: &str,
        headers: &HeaderMap,
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

    /// The request `post_json` sends.
    pub fn request(
        &self,
        url: &str,
        headers: &HeaderMap,
        body: &Value,
    ) -> reqwest::Result<reqwest::Request> {
        self.http
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .headers(headers.clone())
            .json(body)
            .build()
    }

    async fn attempt(
        &self,
        url: &str,
        headers: &HeaderMap,
        body: &Value,
    ) -> Result<Value, Failure> {
        self.attempt_unredacted(url, headers, body)
            .await
            .map_err(|f| match f {
                Failure::Transient(r, e) => Failure::Transient(r, redact(&e, headers)),
                Failure::Permanent(e) => Failure::Permanent(redact(&e, headers)),
            })
    }

    async fn attempt_unredacted(
        &self,
        url: &str,
        headers: &HeaderMap,
        body: &Value,
    ) -> Result<Value, Failure> {
        let provider = self.provider.as_str();
        let req = self
            .request(url, headers, body)
            .map_err(|e| Failure::Permanent(format!("building the request: {e}")))?;
        let started = Instant::now();
        let resp = match self.http.execute(req).await {
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
        let location = resp
            .headers()
            .get(LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(crate::config::display_url);
        let text = resp
            .text()
            .await
            .map_err(|e| Failure::Transient(None, format!("reading response: {e}")))?;
        if status.is_success() {
            return serde_json::from_str::<Value>(&text)
                .map_err(|e| Failure::Transient(None, format!("invalid JSON in response: {e}")));
        }
        let mut detail = format!("{status}: {}", error_detail(&text, request_id.as_deref()));
        if status.is_redirection() {
            detail.push_str(&format!(
                " (redirect to {} not followed; set base_url to the final address)",
                location.as_deref().unwrap_or("an unknown location")
            ));
        }
        if is_quota_exhausted(&text) || !retryable(status) {
            Err(Failure::Permanent(detail))
        } else {
            Err(Failure::Transient(retry_after, detail))
        }
    }
}

/// `text` with the sensitive values of `headers` redacted.
pub fn redact(text: &str, headers: &HeaderMap) -> String {
    crate::config::redact_credentials(
        text,
        headers
            .values()
            .filter(|v| v.is_sensitive())
            .filter_map(|v| v.to_str().ok()),
    )
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
    use crate::config::credential_header;
    use reqwest::header::{AUTHORIZATION, HeaderValue};
    use serde_json::json;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const KEY: &str = "sk-live-0123456789abcdef";

    fn bearer() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            AUTHORIZATION,
            credential_header(&format!("Bearer {KEY}")).unwrap(),
        );
        h
    }

    fn transport(retries: u32) -> Transport {
        Transport::new(Provider::OpenAi, Duration::from_secs(5), retries)
            .unwrap()
            .with_max_backoff(Duration::from_millis(10))
    }

    #[test]
    fn built_requests_keep_the_key_out_of_debug_output() {
        let t = transport(0);
        let req = t
            .request(
                "http://127.0.0.1:1/v1/chat/completions",
                &bearer(),
                &json!({}),
            )
            .unwrap();
        assert_eq!(
            req.headers()[AUTHORIZATION],
            format!("Bearer {KEY}").as_str()
        );
        assert!(req.headers()[AUTHORIZATION].is_sensitive());
        let shown = format!("{req:?} {:?} {t:?}", req.headers());
        assert!(!shown.contains(KEY), "{shown}");
    }

    #[test]
    fn redact_removes_sensitive_values_and_bare_tokens() {
        let mut h = bearer();
        h.insert("x-plain", HeaderValue::from_static("not-a-secret"));
        assert_eq!(
            redact(&format!("got Bearer {KEY}; key {KEY}; not-a-secret"), &h),
            "got [redacted]; key [redacted]; not-a-secret"
        );
    }

    #[tokio::test]
    async fn errors_echoing_the_key_are_redacted() {
        // A proxy that echoes the request headers back in its errors.
        let server = MockServer::start().await;
        let echo =
            json!({"error": {"message": format!("upstream rejected Authorization: Bearer {KEY}")}});
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503).set_body_json(echo.clone()))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(echo))
            .mount(&server)
            .await;
        let err = transport(1)
            .post_json(
                &format!("{}/v1/chat/completions", server.uri()),
                &bearer(),
                &json!({}),
            )
            .await
            .unwrap_err();
        let LlmError::Permanent(msg) = &err else {
            panic!("{err}")
        };
        assert!(msg.contains("401") && msg.contains("[redacted]"), "{msg}");
        assert!(!err.to_string().contains(KEY), "{err}");
    }

    #[tokio::test]
    async fn redirects_are_not_followed() {
        // `x-api-key` and custom headers would survive a cross-host redirect.
        let elsewhere = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .expect(0)
            .mount(&elsewhere)
            .await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(307)
                    .insert_header("location", format!("{}/v1/messages", elsewhere.uri())),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut h = HeaderMap::new();
        h.insert("x-api-key", credential_header(KEY).unwrap());
        let err = transport(3)
            .post_json(&format!("{}/v1/messages", server.uri()), &h, &json!({}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, LlmError::Permanent(ref m) if m.contains("307") && m.contains("not followed")),
            "{err}"
        );
    }

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
