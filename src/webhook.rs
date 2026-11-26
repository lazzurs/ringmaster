//! The ring app's webhook: verify, dedupe, store, queue, answer fast.
//!
//! Nothing here talks to the Mac or to Claude; the worker does that.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::db::{InsertOutcome, NewDelivery, now};
use crate::http::AppState;
use crate::notify::{Kind, Notification};
use crate::signature::{self, SignedHeaders};

pub const H_VERSION: &str = "x-index-webhook-version";
pub const H_TRIGGER: &str = "x-index-trigger";
pub const H_TEST: &str = "x-index-test";
pub const H_DELIVERY: &str = "x-index-delivery";
pub const H_SIGNATURE: &str = "x-index-signature";
pub const H_TIMESTAMP: &str = "x-index-timestamp";

/// Fields of the multipart body.
#[derive(Debug, Default, Clone)]
pub struct Payload {
    pub audio: Option<Bytes>,
    pub audio_content_type: Option<String>,
    pub transcription: Option<String>,
    pub test: bool,
    pub recorded_at_ms: Option<i64>,
    pub client: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("missing or invalid multipart content type")]
    ContentType,
    #[error("malformed multipart body: {0}")]
    Multipart(String),
}

/// Parse the multipart body from the raw bytes already read (they had to be
/// read whole for the signature check).
pub async fn parse_multipart(content_type: &str, body: Bytes) -> Result<Payload, ParseError> {
    let boundary = multer::parse_boundary(content_type).map_err(|_| ParseError::ContentType)?;
    let stream = futures_util::stream::once(async move { Ok::<Bytes, std::io::Error>(body) });
    let mut mp = multer::Multipart::new(stream, boundary);
    let mut p = Payload::default();
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| ParseError::Multipart(e.to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "audio" => {
                p.audio_content_type = field.content_type().map(|m| m.to_string());
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ParseError::Multipart(e.to_string()))?;
                if !data.is_empty() {
                    p.audio = Some(data);
                }
            }
            _ => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| ParseError::Multipart(e.to_string()))?;
                match name.as_str() {
                    "transcription" => {
                        let t = text.trim();
                        if !t.is_empty() {
                            p.transcription = Some(t.to_string());
                        }
                    }
                    "test" => p.test = text.trim().eq_ignore_ascii_case("true"),
                    "recordedAt" => p.recorded_at_ms = text.trim().parse().ok(),
                    "client" => p.client = Some(text.trim().to_string()),
                    // Unknown fields are ignored (forward compatibility).
                    _ => {}
                }
            }
        }
    }
    Ok(p)
}

fn header<'a>(h: &'a HeaderMap, name: &str) -> Option<&'a str> {
    h.get(name).and_then(|v| v.to_str().ok()).map(str::trim)
}

fn reject(status: StatusCode, reason: &str) -> Response {
    metrics::counter!("ringmaster_webhook_requests_total", "result" => reason.replace(' ', "_"))
        .increment(1);
    tracing::warn!(%status, reason, "webhook rejected");
    (status, Json(json!({"error": reason}))).into_response()
}

fn accepted(status: StatusCode, label: &'static str, body: serde_json::Value) -> Response {
    metrics::counter!("ringmaster_webhook_requests_total", "result" => label).increment(1);
    (status, Json(body)).into_response()
}

/// Delivery ids end up in logs and the database; keep them boring.
fn valid_delivery_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 200
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
}

/// Longest the webhook spends storing a delivery. Past it the app gets 503
/// and retries; if the store did finish meanwhile, the retry is answered as
/// a duplicate.
pub const STORE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub async fn handle(State(app): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    match tokio::time::timeout(STORE_TIMEOUT, handle_inner(app, headers, body)).await {
        Ok(r) => r,
        Err(_) => reject(StatusCode::SERVICE_UNAVAILABLE, "storage timed out"),
    }
}

async fn handle_inner(app: Arc<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let cfg = &app.config;

    // 1. Version.
    match header(&headers, H_VERSION) {
        Some(signature::SUPPORTED_VERSION) => {}
        Some(_) => return reject(StatusCode::BAD_REQUEST, "unsupported webhook version"),
        None => return reject(StatusCode::BAD_REQUEST, "missing webhook version"),
    }
    let Some(trigger) = header(&headers, H_TRIGGER).filter(|t| !t.is_empty()) else {
        return reject(StatusCode::BAD_REQUEST, "missing trigger");
    };
    let Some(delivery_id) = header(&headers, H_DELIVERY) else {
        return reject(StatusCode::BAD_REQUEST, "missing delivery id");
    };
    if !valid_delivery_id(delivery_id) {
        return reject(StatusCode::BAD_REQUEST, "invalid delivery id");
    }
    let header_test = header(&headers, H_TEST).is_some_and(|v| v.eq_ignore_ascii_case("true"));

    // 2. Signature over the raw body, before parsing anything in it.
    if let Some(secret) = &app.webhook_secret {
        let sh = SignedHeaders {
            version: signature::SUPPORTED_VERSION,
            timestamp: header(&headers, H_TIMESTAMP),
            signature: header(&headers, H_SIGNATURE),
            delivery_id,
            trigger,
            test: header_test,
        };
        let present = sh.signature.is_some() || sh.timestamp.is_some();
        if (present || cfg.webhook.require_signature)
            && let Err(e) = signature::verify(
                secret.as_bytes(),
                &sh,
                &body,
                now(),
                cfg.webhook.timestamp_tolerance_secs,
            )
        {
            return reject(StatusCode::UNAUTHORIZED, &format!("signature: {e}"));
        }
    } else if cfg.webhook.require_signature {
        return reject(
            StatusCode::INTERNAL_SERVER_ERROR,
            "signing secret not configured",
        );
    }

    // A corrupt database may still take the insert, but nothing stored in
    // it is safe: let the app keep the recording and retry.
    if let Some(why) = app.db.status().refuse_writes() {
        tracing::error!(reason = %why, "refusing a delivery: the database is corrupt");
        return reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable");
    }

    // 3. Dedupe: a repeat delivery is acknowledged and dropped.
    match app.db.delivery_exists(delivery_id).await {
        Ok(true) => {
            return accepted(StatusCode::OK, "duplicate", json!({"status": "duplicate"}));
        }
        Ok(false) => {}
        Err(e) => {
            tracing::error!(error = %e, "database error");
            return reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable");
        }
    }

    // 4. Parse.
    let Some(ct) = header(&headers, "content-type") else {
        return reject(StatusCode::BAD_REQUEST, "missing content type");
    };
    let mut payload = match parse_multipart(ct, body).await {
        Ok(p) => p,
        Err(e) => return reject(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    // Authorisation codes come out of the phone's transcription before it
    // is stored, logged or audited; only what it held is kept.
    let phone_code = match payload.transcription.take() {
        Some(t) if app.codes.enabled() => {
            let scan = app.codes.scan(&t);
            payload.transcription = Some(scan.text);
            scan.summary.contains_code().then_some(scan.summary)
        }
        other => {
            payload.transcription = other;
            None
        }
    };
    let is_test = header_test || payload.test || trigger == "test-event";
    let received_at = now();

    // 5. Test events: log, acknowledge, never run the agent.
    if is_test {
        let d = NewDelivery {
            delivery_id: delivery_id.to_string(),
            received_at,
            trigger: trigger.to_string(),
            is_test: true,
            recorded_at_ms: payload.recorded_at_ms,
            client: payload.client.clone(),
            audio_path: None,
            audio_bytes: payload.audio.as_ref().map(|a| a.len() as i64),
            phone_transcription: payload.transcription.clone(),
            profile: None,
            phone_code: phone_code.clone(),
        };
        return match app.db.insert_delivery(d, false).await {
            Ok(InsertOutcome::Duplicate) => {
                accepted(StatusCode::OK, "duplicate", json!({"status": "duplicate"}))
            }
            Ok(_) => {
                tracing::info!(delivery = delivery_id, trigger, transcription = ?payload.transcription, "test event received");
                let _ = app
                    .db
                    .audit(
                        None,
                        Some(delivery_id),
                        "test_event",
                        json!({"trigger": trigger, "transcription": payload.transcription}),
                    )
                    .await;
                if cfg.webhook.notify_test_events {
                    let notifier = app.notifier.clone();
                    let n = Notification {
                        title: "Ringmaster: test event".into(),
                        body: format!(
                            "Webhook test received{}.",
                            payload
                                .transcription
                                .as_deref()
                                .map(|t| format!(": \"{t}\""))
                                .unwrap_or_default()
                        ),
                        kind: Kind::Info,
                    };
                    // Do not hold up the response on ntfy. The task ends
                    // within 20 s whatever the notifier does.
                    tokio::spawn(async move {
                        let send = notifier.send(&n);
                        match tokio::time::timeout(std::time::Duration::from_secs(20), send).await {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => tracing::warn!(error = %e, "test notification failed"),
                            Err(_) => tracing::warn!("test notification timed out"),
                        }
                    });
                }
                accepted(
                    StatusCode::OK,
                    "test",
                    json!({"status": "test event logged"}),
                )
            }
            Err(e) => {
                tracing::error!(error = %e, "database error");
                reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable")
            }
        };
    }

    // 6. Gesture -> profile.
    let Some(profile) = cfg.profile_for_trigger(trigger) else {
        let _ = app
            .db
            .audit(
                None,
                Some(delivery_id),
                "unmapped_trigger",
                json!({"trigger": trigger}),
            )
            .await;
        return reject(
            StatusCode::UNPROCESSABLE_ENTITY,
            "no profile for this trigger",
        );
    };
    if payload.audio.is_none() && payload.transcription.is_none() {
        return reject(StatusCode::BAD_REQUEST, "no audio and no transcription");
    }

    // 7. Persist audio (atomically), then delivery + job in one transaction.
    let mut audio_path = None;
    let audio_bytes = payload.audio.as_ref().map(|a| a.len() as i64);
    if let Some(audio) = &payload.audio {
        let name = format!(
            "{}.m4a",
            hex::encode(Sha256::digest(delivery_id.as_bytes()))
        );
        let rel = format!("audio/{name}");
        let dir = app.data_dir.join("audio");
        let final_path = dir.join(&name);
        let tmp = dir.join(format!(".{name}.{}.tmp", rand::random::<u64>()));
        let data = audio.clone();
        // A hung disk ties up at most a few threads (the request itself is
        // bounded by STORE_TIMEOUT).
        let Ok(permit) = app.health.io_permits.clone().acquire_owned().await else {
            return reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable");
        };
        let guard = app
            .health
            .blocking
            .start("audio_write", STORE_TIMEOUT.as_secs());
        let write = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            use std::io::Write;
            let _held = (permit, guard);
            std::fs::create_dir_all(&dir)?;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&data)?;
            f.sync_all()?;
            std::fs::rename(&tmp, &final_path)?;
            Ok(())
        })
        .await;
        match write {
            Ok(Ok(())) => audio_path = Some(rel),
            Ok(Err(e)) => {
                tracing::error!(error = %e, "writing audio failed");
                return reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable");
            }
            Err(e) => {
                tracing::error!(error = %e, "audio writer panicked");
                return reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable");
            }
        }
    }
    let d = NewDelivery {
        delivery_id: delivery_id.to_string(),
        received_at,
        trigger: trigger.to_string(),
        is_test: false,
        recorded_at_ms: payload.recorded_at_ms,
        client: payload.client.clone(),
        audio_path,
        audio_bytes,
        phone_transcription: payload.transcription.clone(),
        profile: Some(profile.to_string()),
        phone_code,
    };
    match app.db.insert_delivery(d, true).await {
        Ok(InsertOutcome::Queued { job_id }) => {
            let _ = app
                .db
                .audit(
                    Some(job_id),
                    Some(delivery_id),
                    "delivery",
                    json!({
                        "trigger": trigger,
                        "profile": profile,
                        "audio_bytes": audio_bytes,
                        "audio_content_type": payload.audio_content_type,
                        "has_phone_transcription": payload.transcription.is_some(),
                        "recorded_at_ms": payload.recorded_at_ms,
                        "client": payload.client,
                    }),
                )
                .await;
            app.wake.notify_one();
            tracing::info!(delivery = delivery_id, job = job_id, profile, "queued");
            accepted(
                StatusCode::ACCEPTED,
                "queued",
                json!({"status": "queued", "job": job_id}),
            )
        }
        Ok(InsertOutcome::Duplicate) => {
            accepted(StatusCode::OK, "duplicate", json!({"status": "duplicate"}))
        }
        Ok(InsertOutcome::Recorded) => {
            accepted(StatusCode::OK, "recorded", json!({"status": "recorded"}))
        }
        Err(e) => {
            tracing::error!(error = %e, "database error");
            reject(StatusCode::SERVICE_UNAVAILABLE, "storage unavailable")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn parses_all_fields() {
        let body = "--XyZ\r\n\
            Content-Disposition: form-data; name=\"audio\"; filename=\"rec-1.m4a\"\r\n\
            Content-Type: audio/mp4\r\n\r\n\
            AUDIOBYTES\r\n\
            --XyZ\r\n\
            Content-Disposition: form-data; name=\"transcription\"\r\n\r\n\
            turn on the lights\r\n\
            --XyZ\r\n\
            Content-Disposition: form-data; name=\"recordedAt\"\r\n\r\n\
            1763900000123\r\n\
            --XyZ\r\n\
            Content-Disposition: form-data; name=\"client\"\r\n\r\n\
            ring\r\n\
            --XyZ\r\n\
            Content-Disposition: form-data; name=\"future\"\r\n\r\n\
            ignored\r\n\
            --XyZ--\r\n";
        let p = parse_multipart("multipart/form-data; boundary=XyZ", Bytes::from(body))
            .await
            .unwrap();
        assert_eq!(p.audio.as_deref(), Some(&b"AUDIOBYTES"[..]));
        assert_eq!(p.audio_content_type.as_deref(), Some("audio/mp4"));
        assert_eq!(p.transcription.as_deref(), Some("turn on the lights"));
        assert_eq!(p.recorded_at_ms, Some(1_763_900_000_123));
        assert_eq!(p.client.as_deref(), Some("ring"));
        assert!(!p.test);
    }

    #[tokio::test]
    async fn rejects_bad_multipart() {
        assert!(matches!(
            parse_multipart("application/json", Bytes::from_static(b"{}")).await,
            Err(ParseError::ContentType)
        ));
        assert!(
            parse_multipart(
                "multipart/form-data; boundary=a",
                Bytes::from_static(b"--a\r\ngarbage")
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn delivery_id_validation() {
        assert!(valid_delivery_id("3f2c9a1e-0b7d-4c1e-9d55-2b1f0e8a7c66"));
        assert!(!valid_delivery_id("../../etc/passwd"));
        assert!(!valid_delivery_id(""));
        assert!(!valid_delivery_id(&"a".repeat(201)));
    }
}
