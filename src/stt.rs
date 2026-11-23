//! Speech to text: whisper.cpp `whisper-server` (primary, then fallback),
//! then the phone's own transcription.

use std::time::Duration;

use anyhow::Result;
use serde::Deserialize;

use crate::config::SttConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    pub text: String,
    /// `primary`, `fallback` or `phone`.
    pub source: &'static str,
}

#[derive(Debug, thiserror::Error)]
pub enum SttError {
    /// No transcript yet, but one may come later (servers unreachable).
    #[error("transcription unavailable: {0}")]
    Transient(String),
    /// Nothing will ever work for this recording.
    #[error("transcription failed: {0}")]
    Permanent(String),
}

#[derive(Clone)]
pub struct Transcriber {
    cfg: SttConfig,
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct InferenceResponse {
    text: Option<String>,
    error: Option<String>,
}

impl Transcriber {
    pub fn new(cfg: SttConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .user_agent(concat!("ringmaster/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { cfg, http })
    }

    /// Run the fallback chain. `wav` is `None` when the delivery had no audio
    /// (or it could not be decoded).
    pub async fn transcribe(
        &self,
        wav: Option<&[u8]>,
        phone: Option<&str>,
    ) -> Result<Transcript, SttError> {
        let mut errors = Vec::new();
        if let Some(wav) = wav {
            for (source, url) in [
                ("primary", self.cfg.primary_url.as_deref()),
                ("fallback", self.cfg.fallback_url.as_deref()),
            ] {
                let Some(url) = url else { continue };
                match self.whisper(url, wav).await {
                    Ok(text) => {
                        metrics::counter!("ringmaster_stt_requests_total", "source" => source, "result" => "ok")
                            .increment(1);
                        return Ok(Transcript { text, source });
                    }
                    Err(e) => {
                        metrics::counter!("ringmaster_stt_requests_total", "source" => source, "result" => "error")
                            .increment(1);
                        tracing::warn!(source, error = %e, "whisper request failed");
                        errors.push(format!("{source}: {e}"));
                    }
                }
            }
        }
        if self.cfg.use_phone_transcription
            && let Some(p) = phone
            && !p.trim().is_empty()
        {
            return Ok(Transcript {
                text: p.trim().to_string(),
                source: "phone",
            });
        }
        if wav.is_none() {
            return Err(SttError::Permanent(
                "no audio and no usable phone transcription".into(),
            ));
        }
        if self.cfg.primary_url.is_none() && self.cfg.fallback_url.is_none() {
            return Err(SttError::Permanent(
                "no STT server configured and no phone transcription".into(),
            ));
        }
        Err(SttError::Transient(errors.join("; ")))
    }

    async fn whisper(&self, base: &str, wav: &[u8]) -> Result<String> {
        let url = format!(
            "{}/{}",
            base.trim_end_matches('/'),
            self.cfg.inference_path.trim_start_matches('/')
        );
        let file = reqwest::multipart::Part::bytes(wav.to_vec())
            .file_name("recording.wav")
            .mime_str("audio/wav")?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("response_format", "json")
            .text("temperature", "0.0")
            .text("temperature_inc", "0.2")
            .text("language", self.cfg.language.clone());
        if let Some(prompt) = self.cfg.initial_prompt() {
            form = form.text("prompt", prompt);
        }
        let resp = self.http.post(&url).multipart(form).send().await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            anyhow::bail!("HTTP {status}: {}", truncate(&body, 200));
        }
        let parsed: InferenceResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow::anyhow!("bad JSON from whisper ({e}): {}", truncate(&body, 200))
        })?;
        if let Some(err) = parsed.error {
            anyhow::bail!("whisper error: {err}");
        }
        Ok(clean(parsed.text.unwrap_or_default()))
    }
}

/// whisper output often has leading spaces, newlines between segments and
/// non-speech markers such as `[BLANK_AUDIO]`.
fn clean(text: String) -> String {
    let mut out = String::new();
    for word in text.split_whitespace() {
        if word.starts_with('[') && word.ends_with(']') {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(primary: Option<&MockServer>, fallback: Option<&MockServer>) -> SttConfig {
        SttConfig {
            primary_url: primary.map(|s| s.uri()),
            fallback_url: fallback.map(|s| s.uri()),
            vocabulary: vec!["Homey".into(), "Inbox".into()],
            timeout_secs: 5,
            ..Default::default()
        }
    }

    const WAV: &[u8] = b"RIFF....WAVEfmt fake";

    #[tokio::test]
    async fn primary_success_sends_wav_and_prompt() {
        let primary = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/inference"))
            .and(body_string_contains(
                "name=\"file\"; filename=\"recording.wav\"",
            ))
            .and(body_string_contains("RIFF"))
            .and(body_string_contains("Homey, Inbox"))
            .and(body_string_contains("name=\"response_format\""))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"text": " Turn on the\n kitchen lights [BLANK_AUDIO]\n"}),
            ))
            .expect(1)
            .mount(&primary)
            .await;
        let t = Transcriber::new(cfg(Some(&primary), None)).unwrap();
        let out = t.transcribe(Some(WAV), Some("phone text")).await.unwrap();
        assert_eq!(out.text, "Turn on the kitchen lights");
        assert_eq!(out.source, "primary");
    }

    #[tokio::test]
    async fn falls_back_to_second_server_then_phone() {
        let primary = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("{\"error\":\"boom\"}"))
            .mount(&primary)
            .await;
        let fallback = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"text": "from fallback"})),
            )
            .expect(1)
            .mount(&fallback)
            .await;
        let t = Transcriber::new(cfg(Some(&primary), Some(&fallback))).unwrap();
        let out = t.transcribe(Some(WAV), Some("phone text")).await.unwrap();
        assert_eq!(
            out,
            Transcript {
                text: "from fallback".into(),
                source: "fallback"
            }
        );

        // Both servers down: the phone's transcription wins.
        let dead = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&dead)
            .await;
        let t = Transcriber::new(cfg(Some(&primary), Some(&dead))).unwrap();
        let out = t.transcribe(Some(WAV), Some(" phone text ")).await.unwrap();
        assert_eq!(
            out,
            Transcript {
                text: "phone text".into(),
                source: "phone"
            }
        );

        // ...and without one the job must wait for the Mac.
        let err = t.transcribe(Some(WAV), None).await.unwrap_err();
        assert!(matches!(err, SttError::Transient(_)), "{err}");
    }

    #[tokio::test]
    async fn unreachable_server_is_transient() {
        // Nothing listens on this port once the server is dropped.
        let gone = MockServer::start().await;
        let url = gone.uri();
        drop(gone);
        let t = Transcriber::new(SttConfig {
            primary_url: Some(url),
            connect_timeout_secs: 1,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            t.transcribe(Some(WAV), None).await,
            Err(SttError::Transient(_))
        ));
    }

    #[tokio::test]
    async fn no_audio_uses_phone_or_fails_permanently() {
        let t = Transcriber::new(SttConfig::default()).unwrap();
        assert_eq!(
            t.transcribe(None, Some("hi")).await.unwrap().source,
            "phone"
        );
        assert!(matches!(
            t.transcribe(None, None).await,
            Err(SttError::Permanent(_))
        ));
    }
}
