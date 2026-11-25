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

use ringmaster::config::{Config, credential_header};
use ringmaster::db::Db;
use ringmaster::http::{AppState, router};
use ringmaster::llm::Provider;
use ringmaster::llm::transport::Transport;
use ringmaster::notify::LogNotifier;

const KEY: &str = "sk-live-0123456789abcdef";
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
}
