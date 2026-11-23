//! End-to-end tests: signed webhook -> queue -> whisper (mock) -> Claude
//! (wiremock) -> MCP tools (a real rmcp server on localhost) -> notification,
//! including the confirm/deny flow.
//!
//! Run with proxy variables unset if your shell has them.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerConfig};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData as McpError, ServerHandler, schemars, tool, tool_handler, tool_router};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as MockRequest, Respond, ResponseTemplate};

use ringmaster::agent::Agent;
use ringmaster::claude::ClaudeClient;
use ringmaster::config::{Config, McpServerConfig};
use ringmaster::db::{Db, JobStatus};
use ringmaster::http::{AppState, router};
use ringmaster::notify::{Kind, Notification, Notifier};
use ringmaster::signature;
use ringmaster::stt::Transcriber;
use ringmaster::tools::McpBackend;
use ringmaster::worker::Worker;

const SECRET: &str = "0123456789abcdef0123456789abcdef-test";
const TONE: &[u8] = include_bytes!("fixtures/tone-1s.m4a");

// ------------------------------------------------------------ mock MCP server

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct LightArgs {
    /// Light name
    name: String,
    on: bool,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct DoorArgs {
    door: String,
}

#[derive(Clone)]
struct Home {
    calls: Arc<Mutex<Vec<String>>>,
}

#[tool_router]
impl Home {
    #[tool(description = "Turn a light on or off")]
    async fn set_light(
        &self,
        Parameters(a): Parameters<LightArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("set_light {} {}", a.name, a.on));
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "{} is now {}",
            a.name,
            if a.on { "on" } else { "off" }
        ))]))
    }

    #[tool(description = "Unlock a door")]
    async fn unlock_door(
        &self,
        Parameters(a): Parameters<DoorArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("unlock_door {}", a.door));
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "unlocked",
        )]))
    }

    #[tool(description = "Delete every device")]
    async fn delete_everything(&self) -> Result<CallToolResult, McpError> {
        self.calls.lock().unwrap().push("delete_everything".into());
        Ok(CallToolResult::success(vec![ContentBlock::text("gone")]))
    }
}

#[tool_handler]
impl ServerHandler for Home {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}

async fn start_mcp() -> (String, Arc<Mutex<Vec<String>>>, CancellationToken) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let home = Home {
        calls: calls.clone(),
    };
    let ct = CancellationToken::new();
    let service = StreamableHttpService::new(
        move || Ok(home.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(ct.child_token()),
    );
    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stop = ct.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { stop.cancelled().await })
            .await
            .unwrap();
    });
    (format!("http://{addr}/mcp"), calls, ct)
}

// ------------------------------------------------------------- mock Claude

/// Scripted Claude: returns `turns[n]` for the n-th request and records every
/// request body.
#[derive(Clone)]
struct ScriptedClaude {
    turns: Arc<Vec<Value>>,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl Respond for ScriptedClaude {
    fn respond(&self, req: &MockRequest) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let mut seen = self.seen.lock().unwrap();
        let n = seen.len();
        seen.push(body);
        match self.turns.get(n) {
            Some(t) => ResponseTemplate::new(200).set_body_json(t.clone()),
            None => ResponseTemplate::new(500).set_body_string("script exhausted"),
        }
    }
}

fn tool_use_turn(id: &str, name: &str, input: Value) -> Value {
    json!({
        "id": format!("msg_{id}"),
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "", "signature": format!("sig-{id}")},
            {"type": "tool_use", "id": id, "name": name, "input": input}
        ],
        "usage": {"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 800}
    })
}

fn final_turn(text: &str) -> Value {
    json!({
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 850, "cache_creation_input_tokens": 60}
    })
}

// ------------------------------------------------------------- notifier

#[derive(Default)]
struct Recorder(Mutex<Vec<Notification>>);

#[async_trait]
impl Notifier for Recorder {
    async fn send(&self, n: &Notification) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(n.clone());
        Ok(())
    }
}

impl Recorder {
    fn take(&self) -> Vec<Notification> {
        std::mem::take(&mut self.0.lock().unwrap())
    }
}

// ------------------------------------------------------------- harness

struct Harness {
    app: axum::Router,
    worker: Arc<Worker>,
    db: Db,
    notes: Arc<Recorder>,
    claude_seen: Arc<Mutex<Vec<Value>>>,
    mcp_calls: Arc<Mutex<Vec<String>>>,
    _mcp_stop: CancellationToken,
    _servers: Vec<MockServer>,
    _dir: tempfile::TempDir,
}

const CONFIG: &str = r#"
[server]
public_base_url = "https://ring.test"

[stt]
vocabulary = ["kitchen"]

[claude]
model = "claude-opus-5-5"
effort = "low"
max_retries = 0

[agent]
timezone = "Europe/Dublin"
retry_base_secs = 1

[[tool_policies]]
match = "home/*delete*"
policy = "deny"

[[tool_policies]]
match = "home/unlock_*"
policy = "confirm"

[[tool_policies]]
match = "home/*"
policy = "allow"

[profiles.command]
triggers = ["double-click-hold"]
tools = ["*/*"]
system_prompt = "Command mode."

[profiles.capture]
triggers = ["single-click-hold"]
tools = ["home/set_light"]
system_prompt = "Capture mode."
"#;

async fn harness(turns: Vec<Value>, whisper_text: Option<&str>) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let (mcp_url, mcp_calls, mcp_stop) = start_mcp().await;

    let whisper = MockServer::start().await;
    match whisper_text {
        Some(t) => {
            Mock::given(method("POST"))
                .and(path("/inference"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": t})))
                .mount(&whisper)
                .await
        }
        None => {
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(503))
                .mount(&whisper)
                .await
        }
    }

    let claude_server = MockServer::start().await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ScriptedClaude {
            turns: Arc::new(turns),
            seen: seen.clone(),
        })
        .mount(&claude_server)
        .await;

    let mut config = Config::from_toml(CONFIG).unwrap();
    config.server.data_dir = dir.path().to_path_buf();
    config.stt.primary_url = Some(whisper.uri());
    config.claude.api_base = claude_server.uri();
    config.mcp_servers = vec![McpServerConfig {
        name: "home".into(),
        url: mcp_url,
        token_env: None,
    }];
    let config = Arc::new(config);

    let db = Db::open(&dir.path().join("test.db")).unwrap();
    let notes = Arc::new(Recorder::default());
    let notifier: Arc<dyn Notifier> = notes.clone();
    let claude = ClaudeClient::new(
        &claude_server.uri(),
        "sk-test".into(),
        Duration::from_secs(10),
        0,
    )
    .unwrap();
    let tools = Arc::new(McpBackend::new(
        config.mcp_servers.clone(),
        Default::default(),
        Duration::from_secs(10),
    ));
    let agent =
        Arc::new(Agent::new(config.clone(), db.clone(), claude, tools, notifier.clone()).unwrap());
    let wake = Arc::new(Notify::new());
    let worker = Arc::new(Worker {
        config: config.clone(),
        db: db.clone(),
        transcriber: Transcriber::new(config.stt.clone()).unwrap(),
        agent,
        notifier: notifier.clone(),
        wake: wake.clone(),
        data_dir: dir.path().to_path_buf(),
    });
    let state = Arc::new(AppState {
        config: config.clone(),
        db: db.clone(),
        notifier,
        wake,
        data_dir: dir.path().to_path_buf(),
        webhook_secret: Some(SECRET.into()),
        metrics: None,
    });
    Harness {
        app: router(state),
        worker,
        db,
        notes,
        claude_seen: seen,
        mcp_calls,
        _mcp_stop: mcp_stop,
        _servers: vec![whisper, claude_server],
        _dir: dir,
    }
}

fn multipart(audio: Option<&[u8]>, transcription: Option<&str>, test: bool) -> (String, Vec<u8>) {
    let boundary = "ringmaster-test-boundary";
    let mut body = Vec::new();
    if let Some(a) = audio {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"rec.m4a\"\r\nContent-Type: audio/mp4\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(a);
        body.extend_from_slice(b"\r\n");
    }
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    };
    if let Some(t) = transcription {
        field("transcription", t);
    }
    if test {
        field("test", "true");
    }
    field(
        "recordedAt",
        &(jiff::Timestamp::now().as_millisecond()).to_string(),
    );
    field("client", "ring");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn signed_request(
    delivery: &str,
    trigger: &str,
    test: bool,
    ct: &str,
    body: Vec<u8>,
    ts: i64,
) -> Request<Body> {
    let ts = ts.to_string();
    let sig = signature::sign(SECRET.as_bytes(), &ts, delivery, trigger, test, &body);
    let mut b = Request::post("/webhook")
        .header("content-type", ct)
        .header("x-index-webhook-version", "1")
        .header("x-index-trigger", trigger)
        .header("x-index-delivery", delivery)
        .header("x-index-timestamp", ts)
        .header("x-index-signature", sig)
        .header("x-audio-size", body.len().to_string());
    if test {
        b = b.header("x-index-test", "true");
    }
    b.body(Body::from(body)).unwrap()
}

async fn send(h: &Harness, req: Request<Body>) -> (StatusCode, String) {
    let resp = h.app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn now() -> i64 {
    jiff::Timestamp::now().as_second()
}

async fn tick(h: &Harness) {
    h.worker.tick(&CancellationToken::new()).await;
}

// ------------------------------------------------------------- tests

#[tokio::test]
async fn command_runs_tool_and_reports_result() {
    let h = harness(
        vec![
            tool_use_turn(
                "toolu_1",
                "home__set_light",
                json!({"name": "kitchen", "on": true}),
            ),
            final_turn("Kitchen lights on."),
        ],
        Some(" Turn on the kitchen lights."),
    )
    .await;

    let (ct, body) = multipart(Some(TONE), Some("turn on the kitten lights"), false);
    let (status, text) = send(
        &h,
        signed_request(
            "rec-1",
            "double-click-hold",
            false,
            &ct,
            body.clone(),
            now(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
    assert!(text.contains("queued"));

    // The phone retries: acknowledged, not reprocessed.
    let (status, text) = send(
        &h,
        signed_request("rec-1", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("duplicate"));

    tick(&h).await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert_eq!(notes[0].body, "Kitchen lights on.");

    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let first = &seen[0];
    // Request shape for Opus 5.5.
    assert_eq!(first["model"], "claude-opus-5-5");
    assert_eq!(first["output_config"]["effort"], "low");
    assert_eq!(first["fallbacks"], "default");
    assert_eq!(first["cache_control"]["type"], "ephemeral");
    assert_eq!(first["system"][0]["cache_control"]["type"], "ephemeral");
    for forbidden in ["thinking", "tool_choice", "temperature", "top_p", "top_k"] {
        assert!(
            first.get(forbidden).is_none(),
            "{forbidden} must not be sent"
        );
    }
    // Tools: sorted, denied ones hidden.
    let names: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["home__set_light", "home__unlock_door"]);
    // Volatile context is in the first user message, not the system prompt.
    let user_text = first["messages"][0]["content"][0]["text"].as_str().unwrap();
    assert!(
        user_text.contains("Turn on the kitchen lights."),
        "{user_text}"
    );
    assert!(user_text.contains("Transcript source: primary"));
    assert!(user_text.contains("Current time:"));
    assert!(
        !first["system"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Current time")
    );

    let second = &seen[1];
    // Prefix unchanged: same system and tools, history appended.
    assert_eq!(second["system"], first["system"]);
    assert_eq!(second["tools"], first["tools"]);
    assert_eq!(second["messages"][0], first["messages"][0]);
    // Assistant turn echoed back unchanged, thinking block included.
    assert_eq!(second["messages"][1]["role"], "assistant");
    assert_eq!(
        second["messages"][1]["content"][0]["signature"],
        "sig-toolu_1"
    );
    // All tool results in one user message.
    let results = &second["messages"][2];
    assert_eq!(results["role"], "user");
    assert_eq!(results["content"][0]["type"], "tool_result");
    assert_eq!(results["content"][0]["tool_use_id"], "toolu_1");
    assert_eq!(results["content"][0]["content"], "kitchen is now on");
    assert!(results["content"][0].get("is_error").is_none());

    // Audit trail.
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Done);
    assert_eq!(job.transcript_source.as_deref(), Some("primary"));
    let kinds: Vec<String> =
        h.db.audit_entries(1)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
    for k in [
        "delivery",
        "transcript",
        "agent_start",
        "claude_response",
        "tool_call",
        "job_finished",
    ] {
        assert!(kinds.iter().any(|x| x == k), "missing audit {k}: {kinds:?}");
    }
}

#[tokio::test]
async fn confirm_approve_resumes_and_runs_tool() {
    let h = harness(
        vec![
            tool_use_turn("toolu_9", "home__unlock_door", json!({"door": "front"})),
            final_turn("Front door unlocked."),
        ],
        Some("unlock the front door"),
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-2", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);

    tick(&h).await;
    assert!(
        h.mcp_calls.lock().unwrap().is_empty(),
        "must not run before approval"
    );
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::AwaitingConfirmation);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    let Kind::Confirm {
        approve_url: Some(approve),
        view_url: Some(view),
        ..
    } = &notes[0].kind
    else {
        panic!("expected confirm notification: {:?}", notes[0]);
    };
    assert!(notes[0].body.contains("home/unlock_door"));
    let approve_path = approve
        .strip_prefix("https://ring.test")
        .unwrap()
        .to_string();
    let view_path = view.strip_prefix("https://ring.test").unwrap().to_string();

    // GET shows the page and changes nothing.
    let (status, page) = send(&h, Request::get(&view_path).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(page.contains("home/unlock_door"));
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );

    let (status, _) = send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // One-time token.
    let (status, _) = send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::GONE);

    tick(&h).await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door front".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.last().unwrap().body, "Front door unlocked.");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1]["messages"][2]["content"][0]["content"], "unlocked");
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::Done
    );
}

#[tokio::test]
async fn confirm_deny_returns_error_result_without_running() {
    let h = harness(
        vec![
            tool_use_turn("toolu_5", "home__unlock_door", json!({"door": "back"})),
            final_turn("OK, I left the back door locked."),
        ],
        Some("unlock the back door"),
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-3", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;
    let notes = h.notes.take();
    let Kind::Confirm {
        deny_url: Some(deny),
        ..
    } = &notes[0].kind
    else {
        panic!()
    };
    let deny_path = deny.strip_prefix("https://ring.test").unwrap().to_string();
    let (status, _) = send(&h, Request::post(&deny_path).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);

    tick(&h).await;
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let seen = h.claude_seen.lock().unwrap().clone();
    let result = &seen[1]["messages"][2]["content"][0];
    assert_eq!(result["is_error"], true);
    assert!(result["content"].as_str().unwrap().contains("declined"));
    assert_eq!(
        h.notes.take().last().unwrap().body,
        "OK, I left the back door locked."
    );
}

#[tokio::test]
async fn profile_allow_list_limits_tools() {
    let h = harness(vec![final_turn("Added.")], Some("note: buy milk")).await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-4", "single-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    tick(&h).await;
    let seen = h.claude_seen.lock().unwrap().clone();
    let names: Vec<&str> = seen[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["home__set_light"]);
    assert!(
        seen[0]["system"][0]["text"]
            .as_str()
            .unwrap()
            .ends_with("Capture mode.")
    );
}

#[tokio::test]
async fn whisper_down_falls_back_to_phone_transcription() {
    let h = harness(vec![final_turn("Done it.")], None).await;
    let (ct, body) = multipart(Some(TONE), Some("lights off"), false);
    send(
        &h,
        signed_request("rec-5", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.transcript_source.as_deref(), Some("phone"));
    assert_eq!(job.transcript.as_deref(), Some("lights off"));
    assert_eq!(job.status, JobStatus::Done);
}

#[tokio::test]
async fn whisper_down_without_phone_text_retries_later() {
    let h = harness(vec![final_turn("never")], None).await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-6", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Queued);
    assert!(h.db.next_due().await.unwrap().unwrap() > now() - 1);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].kind, Kind::Error);
    assert!(notes[0].body.contains("keep trying"), "{}", notes[0].body);
    assert!(h.claude_seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn test_event_is_logged_not_run() {
    let h = harness(vec![], Some("x")).await;
    let (ct, body) = multipart(None, Some("This is a test"), true);
    let (status, text) = send(
        &h,
        signed_request("test-1", "test-event", true, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(text.contains("test event"));
    tick(&h).await;
    assert!(h.db.get_job(1).await.unwrap().is_none());
    assert!(h.claude_seen.lock().unwrap().is_empty());
    // The notification is sent in the background.
    for _ in 0..50 {
        if !h.notes.0.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(h.notes.take()[0].kind, Kind::Info);
}

#[tokio::test]
async fn rejects_bad_signature_stale_timestamp_and_unknown_version() {
    let h = harness(vec![], Some("x")).await;
    let (ct, body) = multipart(None, Some("hi"), false);

    let mut req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    req.headers_mut()
        .insert("x-index-signature", "0".repeat(64).parse().unwrap());
    assert_eq!(send(&h, req).await.0, StatusCode::UNAUTHORIZED);

    let req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now() - 301,
    );
    assert_eq!(send(&h, req).await.0, StatusCode::UNAUTHORIZED);

    let mut req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    req.headers_mut()
        .insert("x-index-webhook-version", "2".parse().unwrap());
    assert_eq!(send(&h, req).await.0, StatusCode::BAD_REQUEST);

    let mut req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    req.headers_mut().remove("x-index-signature");
    req.headers_mut().remove("x-index-timestamp");
    assert_eq!(send(&h, req).await.0, StatusCode::UNAUTHORIZED);

    // Nothing was stored by the rejected attempts; a good one still works.
    let req = signed_request(
        "rec-7",
        "double-click-hold",
        false,
        &ct,
        body.clone(),
        now(),
    );
    assert_eq!(send(&h, req).await.0, StatusCode::ACCEPTED);

    // Unmapped gesture.
    let req = signed_request("rec-8", "triple-click", false, &ct, body, now());
    assert_eq!(send(&h, req).await.0, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn health_endpoint() {
    let h = harness(vec![], Some("x")).await;
    let (status, body) = send(&h, Request::get("/healthz").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "ok");
}
