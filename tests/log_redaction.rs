//! Secrets stay out of the logs, at every level and from every crate
//! (reqwest, hyper and h2 included).
//!
//! This file holds a single test on purpose: tracing caches per-callsite
//! interest, and a scoped subscriber in a binary with tests running in
//! parallel can miss events. Here the process-wide subscriber is ours alone.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde_json::json;
use tokio::sync::Notify;
use tower::ServiceExt;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use ringmaster::agent::Agent;
use ringmaster::audio::Decoder;
use ringmaster::authcode::AuthCodes;
use ringmaster::config::{Config, credential_header};
use ringmaster::db::Db;
use ringmaster::http::{AppState, router};
use ringmaster::llm::transport::Transport;
use ringmaster::llm::{Llms, Provider};
use ringmaster::notify::LogNotifier;
use ringmaster::signature;
use ringmaster::stt::Transcriber;
use ringmaster::tools::McpBackend;
use ringmaster::worker::Worker;
use tokio_util::sync::CancellationToken;

const KEY: &str = "sk-live-0123456789abcdef";
const SECRET: &str = "0123456789abcdef0123456789abcdef-test";
const TONE: &[u8] = include_bytes!("fixtures/tone-1s.m4a");
const TOKEN: &str = "c0nf1rm4t10nt0k3n0123456789abcdef0123456789abcdef";

#[derive(Clone)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn take(buf: &Buf) -> String {
    String::from_utf8(std::mem::take(&mut *buf.0.lock().unwrap())).unwrap()
}

#[tokio::test]
async fn secrets_stay_out_of_logs() {
    let buf = Buf(Arc::new(Mutex::new(Vec::new())));
    let writer = buf.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .init();

    // 1. A failing provider call, retried, from a server that echoes the
    // key back in its error bodies.
    let server = MockServer::start().await;
    let echo = json!({"error": {"message": format!("rejected Authorization: Bearer {KEY}")}});
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_json(echo.clone()))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_json(echo))
        .mount(&server)
        .await;
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        credential_header(&format!("Bearer {KEY}")).unwrap(),
    );
    let transport = Transport::new(Provider::OpenAi, Duration::from_secs(5), 1)
        .unwrap()
        .with_max_backoff(Duration::from_millis(10));
    let err = transport
        .post_json(
            &format!("{}/v1/chat/completions", server.uri()),
            &headers,
            &json!({"model": "m"}),
        )
        .await
        .unwrap_err();
    tracing::error!(error = %err, ?headers, "job failed");

    let logs = take(&buf);
    assert!(logs.contains("retrying LLM request"), "{logs}");
    assert!(
        logs.contains("hyper"),
        "dependency logs not captured: {logs}"
    );
    assert!(logs.contains("[redacted]"), "{logs}");
    assert!(!logs.contains(KEY), "{logs}");

    // 2. Confirmation links: the token never reaches the request span.
    let dir = tempfile::tempdir().unwrap();
    let config = Arc::new(Config::from_toml("").unwrap());
    let state = Arc::new(AppState {
        health: Arc::new(ringmaster::health::Health::new(&config, 0)),
        config,
        db: Db::open(&dir.path().join("t.db")).unwrap(),
        notifier: Arc::new(LogNotifier),
        wake: Arc::new(Notify::new()),
        data_dir: dir.path().to_path_buf(),
        webhook_secret: None,
        metrics: None,
        codes: Arc::new(AuthCodes::disabled()),
    });
    for (m, uri) in [
        ("GET", format!("/confirm/{TOKEN}")),
        ("POST", format!("/confirm/{TOKEN}/approve")),
    ] {
        let resp = router(state.clone())
            .oneshot(
                Request::builder()
                    .method(m)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
    let logs = take(&buf);
    assert!(logs.contains("/confirm/{token}"), "{logs}");
    assert!(!logs.contains(TOKEN), "{logs}");

    // 3. Spoken authorisation codes: the webhook (test events log their
    // transcription) and a whole job, from whisper to the notification,
    // never log the code, at any level.
    let whisper = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({"text": "Disarm the alarm, authorisation code Delta Beta five."}),
        ))
        .mount(&whisper)
        .await;
    let llm = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Nothing to do."}],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&llm)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_toml(
        "[auth_codes.codes.alarm]\ncode_env = \"LOG_TEST_CODE\"\n\
         [profiles.command]\ntriggers = [\"double-click-hold\"]\nsystem_prompt = \"x\"\n",
    )
    .unwrap();
    config.server.data_dir = dir.path().to_path_buf();
    config.stt.primary_url = Some(whisper.uri());
    config.claude.api_base = llm.uri();
    let config = Arc::new(config);
    let codes =
        Arc::new(AuthCodes::load(&config.auth_codes, |_| Ok(Some("Delta Beta 5".into()))).unwrap());
    let db = Db::open(&dir.path().join("t.db")).unwrap();
    let health = Arc::new(ringmaster::health::Health::new(&config, 0));
    let keys = [(Provider::Anthropic, "sk-test".to_string())].into();
    let agent = Arc::new(
        Agent::new(
            config.clone(),
            db.clone(),
            Llms::new(&config, &keys).unwrap(),
            Arc::new(McpBackend::new(
                vec![],
                Default::default(),
                Duration::from_secs(5),
            )),
            Arc::new(LogNotifier),
        )
        .unwrap()
        .with_auth_codes(codes.clone()),
    );
    let worker = Arc::new(Worker {
        config: config.clone(),
        db: db.clone(),
        transcriber: Transcriber::new(config.stt.clone()).unwrap(),
        agent,
        notifier: Arc::new(LogNotifier),
        wake: Arc::new(Notify::new()),
        data_dir: dir.path().to_path_buf(),
        health: health.clone(),
        decoder: Decoder::new(&config.stt, health.blocking.clone()),
    });
    let app = router(Arc::new(AppState {
        config: config.clone(),
        db,
        notifier: Arc::new(LogNotifier),
        wake: Arc::new(Notify::new()),
        data_dir: dir.path().to_path_buf(),
        webhook_secret: Some(SECRET.into()),
        metrics: None,
        health,
        codes,
    }));
    for (delivery, trigger, test) in [
        ("log-test", "test-event", true),
        ("log-job", "double-click-hold", false),
    ] {
        let boundary = "b";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"r.m4a\"\r\nContent-Type: audio/mp4\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(TONE);
        body.extend_from_slice(
            format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"transcription\"\r\n\r\nDisarm the alarm, auth code delta beta 5\r\n--{boundary}--\r\n").as_bytes(),
        );
        let ts = jiff::Timestamp::now().as_second().to_string();
        let sig = signature::sign(SECRET.as_bytes(), &ts, delivery, trigger, test, &body);
        let mut req = Request::post("/webhook")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .header("x-index-webhook-version", "1")
            .header("x-index-trigger", trigger)
            .header("x-index-delivery", delivery)
            .header("x-index-timestamp", ts)
            .header("x-index-signature", sig);
        if test {
            req = req.header("x-index-test", "true");
        }
        let resp = app
            .clone()
            .oneshot(req.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        assert!(resp.status().is_success(), "{}", resp.status());
    }
    worker.tick(&CancellationToken::new()).await;
    let logs = take(&buf);
    assert!(logs.contains("test event received"), "{logs}");
    assert!(logs.contains("Nothing to do."), "the job ran: {logs}");
    assert!(logs.contains("[authorisation code accepted]"), "{logs}");
    let lower = logs.to_lowercase();
    for word in ["delta", "beta"] {
        assert!(!lower.contains(word), "{word} in the logs: {logs}");
    }
}
