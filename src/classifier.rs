//! Automatic approval of `confirm` tool calls by a local Shieldstral
//! classifier.
//!
//! Shieldstral (Mistral, Apache 2.0) answers one yes/no question about a
//! document. It is served behind an OpenAI-compatible
//! `/v1/chat/completions` endpoint (llama.cpp `llama-server`, vLLM, SGLang)
//! and asked for a single token with its log-probabilities; the score is the
//! softmax of the best `yes` and `no` log-probabilities, as in the model
//! card's reference helper. Here the document is the spoken request plus
//! one proposed tool call, and "yes" means "do not run this without asking".
//!
//! The classifier can approve but never deny. Any failure to get a clean
//! verdict (server down, timeout, an error status, a reply without `yes`/
//! `no` log-probabilities, a call too long to judge) is an error, and the
//! caller then asks a human, exactly as without a classifier.
//!
//! Shieldstral was trained on content safety (prompts, responses, images),
//! not on judging agent actions; see the README before relying on it.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{ApprovalConfig, Config, credential_header};
use crate::health::Health;
use crate::llm::LlmError;
use crate::llm::transport::Transport;

/// The fixed system message from the model card.
pub const SYSTEM_PROMPT: &str = "Judge whether the Document meets the requirements based on the \
Query and the Instruction provided. Note that the answer can only be \"yes\" or \"no\".";

/// Token spellings counted as each answer (after trimming and lowercasing),
/// as in the model card's helper.
const YES: [&str; 4] = ["yes", "yes.", "\"yes\"", "'yes'"];
const NO: [&str; 4] = ["no", "no.", "\"no\"", "'no'"];
/// Log-probabilities below this are clamped to it (the helper's start value).
const LOGPROB_FLOOR: f64 = -10.0;
/// Alternatives asked for at the answer position.
pub const TOP_LOGPROBS: u32 = 20;

/// Longest transcript and arguments the classifier judges. A call that does
/// not fit is never approved automatically: cutting it would hide part of
/// what would run.
pub const MAX_TRANSCRIPT_CHARS: usize = 2000;
pub const MAX_ARGUMENTS_CHARS: usize = 4000;
/// Context that may be shortened (it does not change what runs).
const MAX_DESCRIPTION_CHARS: usize = 1000;
const MAX_ASSISTANT_CHARS: usize = 1000;

/// What the classifier makes of a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Below `auto_approve_below` (or `threshold`): may run without asking.
    Safe,
    /// The grey band between `auto_approve_below` and `threshold`.
    Uncertain,
    /// Above `threshold`: flagged.
    Unsafe,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Safe => "safe",
            Verdict::Uncertain => "uncertain",
            Verdict::Unsafe => "unsafe",
        }
    }
}

/// Why there is no verdict. Every one of these sends the call to a human.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ClassifyError {
    #[error("no verdict within {0} s")]
    Timeout(u64),
    /// Connection errors, 429/5xx after retries.
    #[error("classifier unavailable: {0}")]
    Unavailable(String),
    /// A 4xx: a wrong URL, model name or key.
    #[error("classifier rejected the request: {0}")]
    Rejected(String),
    /// A 2xx without usable `yes`/`no` log-probabilities.
    #[error("unusable classifier reply: {0}")]
    Malformed(String),
    /// The call is too long to judge whole.
    #[error("{0}")]
    TooLong(String),
    /// Classifier mode is on but no classifier was built (tests).
    #[error("no classifier configured")]
    NotConfigured,
}

impl ClassifyError {
    /// A fixed word for alerts and metrics (never server text).
    pub fn kind(&self) -> &'static str {
        match self {
            ClassifyError::Timeout(_) => "timeout",
            ClassifyError::Unavailable(_) => "unavailable",
            ClassifyError::Rejected(_) => "rejected",
            ClassifyError::Malformed(_) => "malformed",
            ClassifyError::TooLong(_) => "too_long",
            ClassifyError::NotConfigured => "not_configured",
        }
    }

    /// Whether this says something about the classifier's health (a call
    /// that is too long does not).
    fn is_failure(&self) -> bool {
        !matches!(self, ClassifyError::TooLong(_))
    }
}

/// One proposed tool call and what it is judged against.
#[derive(Debug, Clone)]
pub struct ProposedCall<'a> {
    /// What the user said, verbatim.
    pub transcript: &'a str,
    /// Text the model wrote alongside the call, if any.
    pub assistant_text: Option<&'a str>,
    /// `<server>/<tool>`.
    pub tool: &'a str,
    pub description: &'a str,
    pub arguments: &'a Value,
}

/// The `<Document>` for one call. Errors when the transcript or the
/// arguments are too long to include whole.
pub fn document(call: &ProposedCall<'_>) -> Result<String, ClassifyError> {
    let transcript = call.transcript.trim();
    if transcript.chars().count() > MAX_TRANSCRIPT_CHARS {
        return Err(ClassifyError::TooLong(format!(
            "the transcript is over {MAX_TRANSCRIPT_CHARS} characters"
        )));
    }
    let arguments = serde_json::to_string(call.arguments).unwrap_or_default();
    if arguments.chars().count() > MAX_ARGUMENTS_CHARS {
        return Err(ClassifyError::TooLong(format!(
            "the arguments are over {MAX_ARGUMENTS_CHARS} characters"
        )));
    }
    let mut doc = format!("[User request]\n{transcript}\n\n");
    if let Some(t) = call.assistant_text.map(str::trim).filter(|t| !t.is_empty()) {
        doc.push_str(&format!(
            "[Assistant]\n{}\n\n",
            crate::agent::truncate(t, MAX_ASSISTANT_CHARS)
        ));
    }
    doc.push_str(&format!(
        "[Proposed action]\nTool: {}\nDescription: {}\nArguments: {arguments}",
        call.tool,
        crate::agent::truncate(call.description.trim(), MAX_DESCRIPTION_CHARS),
    ));
    Ok(doc)
}

/// The score in a chat-completions reply: the softmax of the best `yes` and
/// `no` log-probabilities at the first generated position. Also reads
/// llama.cpp's `post_sampling_probs` shape (`top_probs` / `prob`).
pub fn score_from_response(resp: &Value) -> Result<f64, String> {
    let first = resp
        .pointer("/choices/0/logprobs/content/0")
        .ok_or("no logprobs in the reply (does the server support logprobs?)")?;
    let mut candidates: Vec<(&str, f64)> = entry_logprob(first).into_iter().collect();
    for key in ["top_logprobs", "top_probs"] {
        if let Some(list) = first.get(key).and_then(Value::as_array) {
            candidates.extend(list.iter().filter_map(entry_logprob));
        }
    }
    let (mut yes, mut no) = (None::<f64>, None::<f64>);
    for (token, lp) in candidates {
        let t = token.trim().to_lowercase();
        let slot = if YES.contains(&t.as_str()) {
            &mut yes
        } else if NO.contains(&t.as_str()) {
            &mut no
        } else {
            continue;
        };
        *slot = Some(slot.map_or(lp, |v| v.max(lp)));
    }
    if yes.is_none() && no.is_none() {
        return Err("no \"yes\" or \"no\" token among the top logprobs".into());
    }
    let y = yes.unwrap_or(LOGPROB_FLOOR).max(LOGPROB_FLOOR);
    let n = no.unwrap_or(LOGPROB_FLOOR).max(LOGPROB_FLOOR);
    // exp(y) / (exp(y) + exp(n)), without overflow.
    Ok(1.0 / (1.0 + (n - y).exp()))
}

/// The token and log-probability of one logprobs entry (`logprob`, or
/// llama.cpp's `prob` with `post_sampling_probs`).
fn entry_logprob(entry: &Value) -> Option<(&str, f64)> {
    let token = entry.get("token").and_then(Value::as_str)?;
    let lp = match (
        entry.get("logprob").and_then(Value::as_f64),
        entry.get("prob").and_then(Value::as_f64),
    ) {
        (Some(lp), _) => lp,
        (None, Some(p)) if p > 0.0 => p.ln(),
        (None, Some(_)) => f64::NEG_INFINITY,
        (None, None) => return None,
    };
    (!lp.is_nan()).then_some((token, lp))
}

/// Client for the classifier server. The key is held only as a sensitive
/// header value, so `Debug` never shows it.
pub struct Classifier {
    transport: Transport,
    url: String,
    headers: HeaderMap,
    /// A key was set but cannot be sent as a header.
    invalid_key: bool,
    model: String,
    instruct: String,
    query: String,
    threshold: f64,
    cutoff: f64,
    timeout: Duration,
    health: Option<Arc<Health>>,
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `headers` holds the key as a sensitive value (shown as such).
        f.debug_struct("Classifier")
            .field("url", &self.url)
            .field("headers", &self.headers)
            .field("model", &self.model)
            .field("threshold", &self.threshold)
            .field("cutoff", &self.cutoff)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl Classifier {
    pub fn new(cfg: &ApprovalConfig, api_key: Option<String>) -> Result<Self> {
        let base = cfg.base_url.as_deref().unwrap_or_default();
        let base = base.trim_end_matches('/');
        let base = base.strip_suffix("/v1").unwrap_or(base);
        let timeout = Duration::from_secs(cfg.request_timeout_secs);
        let mut headers = HeaderMap::new();
        let mut invalid_key = false;
        if let Some(k) = &api_key {
            match credential_header(&format!("Bearer {k}")) {
                Some(v) => {
                    headers.insert(AUTHORIZATION, v);
                }
                None => invalid_key = true,
            }
        }
        Ok(Self {
            // The overall bound in `classify` is what ends a slow request;
            // the transport's own limit is a backstop just past it.
            transport: Transport::named(
                "classifier",
                timeout + Duration::from_secs(1),
                cfg.max_retries,
            )?
            .with_max_backoff(Duration::from_secs(2)),
            url: format!("{base}/v1/chat/completions"),
            headers,
            invalid_key,
            model: cfg.model.clone(),
            instruct: cfg.instruct().trim().to_string(),
            query: cfg.query().trim().to_string(),
            threshold: cfg.threshold,
            cutoff: cfg.auto_approve_cutoff(),
            timeout,
            health: None,
        })
    }

    /// The classifier the config asks for, if any profile uses one.
    pub fn from_config(config: &Config, api_key: Option<String>) -> Result<Option<Self>> {
        if !config.uses_classifier() {
            return Ok(None);
        }
        Self::new(&config.approval, api_key).map(Some)
    }

    /// Report each verdict and failure to `health` (for the alert).
    pub fn with_health(mut self, health: Arc<Health>) -> Self {
        self.health = Some(health);
        self
    }

    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.transport = self.transport.with_max_backoff(d);
        self
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Scores below this are approved automatically.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    pub fn verdict(&self, score: f64) -> Verdict {
        if score < self.cutoff {
            Verdict::Safe
        } else if score > self.threshold {
            Verdict::Unsafe
        } else {
            Verdict::Uncertain
        }
    }

    /// The request body, as the model card describes it.
    pub fn body(&self, document: &str) -> Value {
        json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format!(
                    "<Instruct>: {}\n\n<Query>: {}\n\n<Document>: {document}",
                    self.instruct, self.query
                )},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": true,
            "top_logprobs": TOP_LOGPROBS,
            "stream": false,
        })
    }

    /// The unsafe score of one proposed call (0 = clearly fine, 1 = clearly
    /// not). Bounded by `approval.request_timeout_secs`, retries included.
    pub async fn classify(&self, call: &ProposedCall<'_>) -> Result<f64, ClassifyError> {
        let result = self.classify_inner(call).await;
        if let Some(h) = &self.health {
            match &result {
                Ok(_) => h.classifier_ok(),
                Err(e) if e.is_failure() => h.classifier_failed(e.kind(), crate::db::now()),
                Err(_) => {}
            }
        }
        result
    }

    async fn classify_inner(&self, call: &ProposedCall<'_>) -> Result<f64, ClassifyError> {
        let doc = document(call)?;
        if self.invalid_key {
            return Err(ClassifyError::Rejected(
                "the key in approval.api_key_env is not a valid header value".into(),
            ));
        }
        let body = self.body(&doc);
        let resp = tokio::time::timeout(
            self.timeout,
            self.transport.post_json(&self.url, &self.headers, &body),
        )
        .await
        .map_err(|_| ClassifyError::Timeout(self.timeout.as_secs()))?
        .map_err(|e| match e {
            LlmError::Transient(m) => ClassifyError::Unavailable(m),
            LlmError::Permanent(m) => ClassifyError::Rejected(m),
        })?;
        score_from_response(&resp).map_err(ClassifyError::Malformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn reply(top: &[(&str, f64)]) -> Value {
        json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": top[0].0},
            "logprobs": {"content": [{"token": top[0].0, "logprob": top[0].1,
                "top_logprobs": top.iter().map(|(t, l)| json!({"token": t, "logprob": l})).collect::<Vec<_>>()}]}}]})
    }

    #[test]
    fn score_is_the_softmax_of_yes_and_no() {
        let s = score_from_response(&reply(&[("no", -0.05), ("yes", -3.0)])).unwrap();
        let want = (-3.0f64).exp() / ((-3.0f64).exp() + (-0.05f64).exp());
        assert!((s - want).abs() < 1e-12, "{s} {want}");
        // Spelling variants count; the best of each wins.
        let s =
            score_from_response(&reply(&[(" Yes", -0.1), ("\"no\"", -2.5), ("No", -2.4)])).unwrap();
        assert!(s > 0.9, "{s}");
        // A missing answer is clamped to the floor, as in the reference.
        let s = score_from_response(&reply(&[("no", -0.001), ("maybe", -9.0)])).unwrap();
        assert!(s < 0.001, "{s}");
        let s = score_from_response(&reply(&[("yes", -30.0), ("no", -40.0)])).unwrap();
        assert!((s - 0.5).abs() < 1e-12, "{s}");
    }

    #[test]
    fn llama_cpp_post_sampling_shape_is_read() {
        let r = json!({"choices": [{"logprobs": {"content": [{"token": "no", "prob": 0.8,
            "top_probs": [{"token": "no", "prob": 0.8}, {"token": "yes", "prob": 0.2}]}]}}]});
        let s = score_from_response(&r).unwrap();
        assert!((s - 0.2).abs() < 1e-9, "{s}");
    }

    #[test]
    fn replies_without_a_usable_answer_are_errors() {
        for bad in [
            json!({}),
            json!({"choices": [{"message": {"content": "no"}}]}),
            json!({"choices": [{"logprobs": {"content": []}}]}),
            reply(&[("Sure", -0.1), ("I", -1.0)]),
        ] {
            assert!(score_from_response(&bad).is_err(), "{bad}");
        }
    }

    fn cfg(url: &str) -> ApprovalConfig {
        ApprovalConfig {
            base_url: Some(url.into()),
            auto_approve_below: Some(0.2),
            ..Default::default()
        }
    }

    #[test]
    fn verdict_bands() {
        let c = Classifier::new(&cfg("http://127.0.0.1:1"), None).unwrap();
        assert_eq!(c.verdict(0.19), Verdict::Safe);
        assert_eq!(c.verdict(0.2), Verdict::Uncertain);
        assert_eq!(c.verdict(0.5), Verdict::Uncertain);
        assert_eq!(c.verdict(0.51), Verdict::Unsafe);
        let plain = Classifier::new(
            &ApprovalConfig {
                base_url: Some("http://127.0.0.1:1/v1/".into()),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(plain.verdict(0.49), Verdict::Safe);
        assert_eq!(plain.verdict(0.5), Verdict::Uncertain);
        assert_eq!(plain.url, "http://127.0.0.1:1/v1/chat/completions");
    }

    fn call<'a>(args: &'a Value, transcript: &'a str) -> ProposedCall<'a> {
        ProposedCall {
            transcript,
            assistant_text: Some("Unlocking the front door."),
            tool: "homey/unlock_door",
            description: "Unlock a door",
            arguments: args,
        }
    }

    #[test]
    fn document_holds_everything_and_refuses_to_cut() {
        let args = json!({"door": "front"});
        let doc = document(&call(&args, " unlock the front door ")).unwrap();
        assert_eq!(
            doc,
            "[User request]\nunlock the front door\n\n[Assistant]\nUnlocking the front door.\n\n\
             [Proposed action]\nTool: homey/unlock_door\nDescription: Unlock a door\n\
             Arguments: {\"door\":\"front\"}"
        );
        let long = json!({"text": "x".repeat(MAX_ARGUMENTS_CHARS)});
        assert!(matches!(
            document(&call(&long, "note")),
            Err(ClassifyError::TooLong(_))
        ));
        let said = "y".repeat(MAX_TRANSCRIPT_CHARS + 1);
        assert!(matches!(
            document(&call(&args, &said)),
            Err(ClassifyError::TooLong(_))
        ));
    }

    #[tokio::test]
    async fn key_is_sent_sensitive_and_redacted_from_errors() {
        const KEY: &str = "shield-key-0123456789";
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", format!("Bearer {KEY}").as_str()))
            .respond_with(
                ResponseTemplate::new(401)
                    .set_body_json(json!({"error": {"message": format!("bad key Bearer {KEY}")}})),
            )
            .mount(&server)
            .await;
        let c = Classifier::new(&cfg(&server.uri()), Some(KEY.into())).unwrap();
        assert!(!format!("{c:?}").contains(KEY));
        let args = json!({});
        let err = c.classify(&call(&args, "hi")).await.unwrap_err();
        assert_eq!(err.kind(), "rejected");
        assert!(!err.to_string().contains(KEY), "{err}");
        assert!(err.to_string().contains("[redacted]"), "{err}");
    }
}
