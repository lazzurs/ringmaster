//! End-to-end tests: signed webhook -> queue -> whisper (mock) -> LLM
//! (wiremock: Claude, OpenAI, Mistral, OpenRouter, Ollama) -> MCP tools (a
//! real rmcp server on localhost) -> notification, including the
//! confirm/deny flow.
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
use wiremock::matchers::{header, method, path, path_regex};
use wiremock::{Mock, MockServer, Request as MockRequest, Respond, ResponseTemplate};

use ringmaster::agent::Agent;
use ringmaster::audio::Decoder;
use ringmaster::authcode::AuthCodes;
use ringmaster::classifier::Classifier;
use ringmaster::config::{Config, McpServerConfig};
use ringmaster::db::{Db, JobStatus};
use ringmaster::health::{AlertBook, Alerter, Health, Watchdog};
use ringmaster::http::{AppState, router};
use ringmaster::llm::{Llms, Provider};
use ringmaster::notify::{Kind, Notification, Notifier};
use ringmaster::signature;
use ringmaster::stt::Transcriber;
use ringmaster::tools::{DiscoveredTool, McpBackend, ToolBackend, ToolError, ToolOutput};
use ringmaster::worker::Worker;

const SECRET: &str = "0123456789abcdef0123456789abcdef-test";
const TONE: &[u8] = include_bytes!("fixtures/tone-1s.m4a");
/// Spoken codes the harness reads "from the environment".
const CODE_ENV: &[(&str, &str)] = &[
    ("RINGMASTER_CODE_ALARM", "Delta Beta 5 | delta better five"),
    ("RINGMASTER_CODE_GARAGE", "echo foxtrot 7"),
];

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

/// Scripted model API: returns `turns[n]` for the n-th request and records
/// every request body, plus its path and headers.
#[derive(Clone)]
struct ScriptedClaude {
    turns: Arc<Vec<Value>>,
    seen: Arc<Mutex<Vec<Value>>>,
    meta: Arc<Mutex<Vec<Value>>>,
}

impl Respond for ScriptedClaude {
    fn respond(&self, req: &MockRequest) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let headers: serde_json::Map<String, Value> = req
            .headers
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), json!(v.to_str().unwrap_or(""))))
            .collect();
        self.meta
            .lock()
            .unwrap()
            .push(json!({"path": req.url.path(), "headers": headers}));
        let mut seen = self.seen.lock().unwrap();
        let n = seen.len();
        seen.push(body);
        match self.turns.get(n) {
            // `{"__status": 400, "__body": {...}}` scripts an API error.
            Some(t) if t.get("__status").is_some() => {
                ResponseTemplate::new(t["__status"].as_u64().unwrap() as u16)
                    .set_body_json(t["__body"].clone())
            }
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
    health: Arc<Health>,
    /// Sends its alerts to `notes` too.
    watchdog: Arc<Watchdog>,
    db: Db,
    notes: Arc<Recorder>,
    claude_seen: Arc<Mutex<Vec<Value>>>,
    /// Path and headers of each model request.
    llm_meta: Arc<Mutex<Vec<Value>>>,
    db_path: std::path::PathBuf,
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
tool_discovery_timeout_secs = 2

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
    harness_with(turns, whisper_text, |home| vec![home]).await
}

/// `servers` gets the live test MCP server ("home") and returns the list the
/// service is configured with.
async fn harness_with(
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
) -> Harness {
    harness_config(CONFIG, turns, whisper_text, servers).await
}

/// The test config with its `[claude]` section replaced by `llm`.
fn config_with(llm: &str) -> String {
    let legacy = "[claude]\nmodel = \"claude-opus-5-5\"\neffort = \"low\"\nmax_retries = 0\n";
    assert!(CONFIG.contains(legacy));
    CONFIG.replace(legacy, llm)
}

async fn harness_config(
    config_text: &str,
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
) -> Harness {
    harness_full(config_text, turns, whisper_text, servers, |t| t).await
}

/// Builds the decoder; tests can swap in a misbehaving one.
type MakeDecoder = Box<dyn FnOnce(&Config, &Health) -> Decoder>;

/// `harness_config`, with the MCP backend wrapped by `wrap_tools`.
async fn harness_full(
    config_text: &str,
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
    wrap_tools: impl FnOnce(Arc<dyn ToolBackend>) -> Arc<dyn ToolBackend>,
) -> Harness {
    harness_decoder(
        config_text,
        turns,
        whisper_text,
        servers,
        wrap_tools,
        Box::new(|c: &Config, h: &Health| Decoder::new(&c.stt, h.blocking.clone())),
    )
    .await
}

async fn harness_decoder(
    config_text: &str,
    turns: Vec<Value>,
    whisper_text: Option<&str>,
    servers: impl FnOnce(McpServerConfig) -> Vec<McpServerConfig>,
    wrap_tools: impl FnOnce(Arc<dyn ToolBackend>) -> Arc<dyn ToolBackend>,
    make_decoder: MakeDecoder,
) -> Harness {
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
    let meta = Arc::new(Mutex::new(Vec::new()));
    // Every provider's endpoint, on one scripted server.
    Mock::given(method("POST"))
        .and(path_regex(r"^/(v1/messages|v1/chat/completions|api/chat)$"))
        .respond_with(ScriptedClaude {
            turns: Arc::new(turns),
            seen: seen.clone(),
            meta: meta.clone(),
        })
        .mount(&claude_server)
        .await;

    let mut config = Config::from_toml(config_text).unwrap();
    config.server.data_dir = dir.path().to_path_buf();
    config.stt.primary_url = Some(whisper.uri());
    config.claude.api_base = claude_server.uri();
    for p in [
        &mut config.llm.openai,
        &mut config.llm.mistral,
        &mut config.llm.openrouter,
        &mut config.llm.ollama,
    ] {
        p.base_url = Some(claude_server.uri());
    }
    config.mcp_servers = servers(McpServerConfig {
        name: "home".into(),
        url: mcp_url,
        ..Default::default()
    });
    let config = Arc::new(config);

    let db = Db::open(&dir.path().join("test.db")).unwrap();
    let notes = Arc::new(Recorder::default());
    let notifier: Arc<dyn Notifier> = notes.clone();
    let keys = [
        (Provider::Anthropic, "sk-test"),
        (Provider::OpenAi, "sk-openai-test"),
        (Provider::Mistral, "mistral-test"),
        (Provider::OpenRouter, "sk-or-test"),
    ]
    .into_iter()
    .map(|(p, k)| (p, k.to_string()))
    .collect();
    let llms = Llms::new(&config, &keys).unwrap();
    let tools = wrap_tools(Arc::new(McpBackend::new(
        config.mcp_servers.clone(),
        Default::default(),
        Duration::from_secs(10),
    )));
    let health = Arc::new(Health::new(&config, now()));
    // The approval classifier, when the config uses one (its base_url is
    // a wiremock server the test started).
    let classifier = Classifier::from_config(&config, None).unwrap().map(|c| {
        Arc::new(
            c.with_health(health.clone())
                .with_max_backoff(Duration::from_millis(10)),
        )
    });
    // Code values as the environment would give them.
    let codes = Arc::new(
        AuthCodes::load(&config.auth_codes, |var| {
            Ok(CODE_ENV
                .iter()
                .find(|(k, _)| *k == var)
                .map(|(_, v)| v.to_string()))
        })
        .unwrap(),
    );
    let agent = Arc::new(
        Agent::new(config.clone(), db.clone(), llms, tools, notifier.clone())
            .unwrap()
            .with_classifier(classifier)
            .with_auth_codes(codes.clone()),
    );
    let wake = Arc::new(Notify::new());
    let worker = Arc::new(Worker {
        config: config.clone(),
        db: db.clone(),
        transcriber: Transcriber::new(config.stt.clone()).unwrap(),
        agent,
        notifier: notifier.clone(),
        wake: wake.clone(),
        data_dir: dir.path().to_path_buf(),
        health: health.clone(),
        decoder: make_decoder(&config, &health),
    });
    let watchdog = Arc::new(Watchdog::new(
        health.clone(),
        db.clone(),
        Alerter::new(notifier.clone(), &config.alerts, AlertBook::load(None)),
    ));
    let state = Arc::new(AppState {
        config: config.clone(),
        db: db.clone(),
        notifier,
        wake,
        data_dir: dir.path().to_path_buf(),
        webhook_secret: Some(SECRET.into()),
        metrics: None,
        health: health.clone(),
        codes,
    });
    Harness {
        app: router(state),
        worker,
        health,
        watchdog,
        db,
        notes,
        claude_seen: seen,
        llm_meta: meta,
        db_path: dir.path().join("test.db"),
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
    // The token is 32 random bytes as unpadded URL-safe base64: 43
    // characters that need no escaping in a path.
    let token = approve_path
        .strip_prefix("/confirm/")
        .and_then(|t| t.strip_suffix("/approve"))
        .unwrap();
    assert_eq!(token.len(), 43, "{token}");
    assert!(
        token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "{token}"
    );
    assert_eq!(view_path, format!("/confirm/{token}"));

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

/// A server nothing is listening on.
fn dead_server(name: &str) -> McpServerConfig {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    McpServerConfig {
        name: name.into(),
        url: format!("http://127.0.0.1:{port}/mcp"),
        ..Default::default()
    }
}

#[tokio::test]
async fn wildcard_profile_skips_unavailable_server() {
    let h = harness_with(
        vec![final_turn("Done.")],
        Some("Turn on the kitchen lights."),
        |home| vec![home, dead_server("homey")],
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-dead-1", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    tick(&h).await;

    // The job ran with the live server's tools and was told what is missing.
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "Claude should have been called once");
    let names: Vec<&str> = seen[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.iter().all(|n| n.starts_with("home__")), "{names:?}");
    let first = seen[0]["messages"][0]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(first.contains("Unavailable right now"), "{first}");
    assert!(first.contains("homey"), "{first}");

    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(notes[0].body.starts_with("Done."), "{}", notes[0].body);
    assert!(
        notes[0].body.contains("Unavailable: homey"),
        "{}",
        notes[0].body
    );
    assert!(
        notes[0].body.contains("Turn on the kitchen lights."),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn only_server_down_still_answers_and_keeps_the_request() {
    // The capture profile names `home/...` and that server is down: the job
    // still runs (with no tools), and the reply carries what the user said.
    let h = harness_with(
        vec![final_turn(
            "I couldn't add that: the task list is unavailable.",
        )],
        Some("buy milk"),
        |_| vec![dead_server("home")],
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, _) = send(
        &h,
        signed_request("rec-dead-2", "single-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    tick(&h).await;

    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].get("tools").is_none(), "no tools to offer");
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Done);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(
        notes[0].body.contains("Unavailable: home"),
        "{}",
        notes[0].body
    );
    assert!(notes[0].body.contains("buy milk"), "{}", notes[0].body);
}

#[tokio::test]
async fn rejected_tool_definitions_fall_back_to_no_tools() {
    let h = harness(
        vec![
            json!({"__status": 400, "__body": {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "tools.0.input_schema: invalid"}
            }}),
            final_turn("Sorry, I can't do that right now."),
        ],
        Some("Turn on the kitchen lights."),
    )
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-400", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    tick(&h).await;

    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "retried once");
    assert!(seen[0].get("tools").is_some());
    assert!(seen[1].get("tools").is_none(), "second try has no tools");
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(
        notes[0].body.contains("Unavailable: all tools"),
        "{}",
        notes[0].body
    );
    assert!(
        notes[0].body.contains("kitchen lights"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn hung_server_is_skipped_quickly() {
    // Accepts connections but never answers.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((sock, _)) = listener.accept().await {
            held.push(sock);
        }
    });
    let h = harness_with(vec![final_turn("Done.")], Some("lights on"), move |home| {
        vec![
            home,
            McpServerConfig {
                name: "slow".into(),
                url: format!("http://127.0.0.1:{port}/mcp"),
                ..Default::default()
            },
        ]
    })
    .await;
    let (ct, body) = multipart(Some(TONE), None, false);
    send(
        &h,
        signed_request("rec-hung", "double-click-hold", false, &ct, body, now()),
    )
    .await;
    let started = std::time::Instant::now();
    tick(&h).await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "took {:?}",
        started.elapsed()
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(
        notes[0].body.contains("Unavailable: slow"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn mcp_requests_carry_configured_headers() {
    use ringmaster::tools::ToolBackend;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("authorization", "Basic cm9iOmFwcC1wYXNz"))
        .and(header("x-extra", "v"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1..)
        .mount(&server)
        .await;
    let backend = McpBackend::new(
        vec![McpServerConfig {
            name: "nc".into(),
            url: format!("{}/mcp", server.uri()),
            ..Default::default()
        }],
        [(
            "nc".to_string(),
            vec![
                (
                    "Authorization".to_string(),
                    "Basic cm9iOmFwcC1wYXNz".to_string(),
                ),
                ("X-Extra".to_string(), "v".to_string()),
            ],
        )]
        .into(),
        Duration::from_secs(5),
    );
    // The mock answers 503, so listing fails; what matters is what was sent.
    assert!(backend.list_tools(&["nc".to_string()]).await.is_err());
    server.verify().await;
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

// ------------------------------------------------- other providers

/// An OpenAI-style reply with one tool call (arguments as a JSON string).
fn chat_tool_turn(id: &str, name: &str, arguments: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "refusal": null,
                "tool_calls": [{"id": id, "type": "function",
                                "function": {"name": name, "arguments": arguments}}]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920}
    })
}

fn chat_final(text: &str) -> Value {
    json!({
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "refusal": null},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 950, "completion_tokens": 8, "total_tokens": 958}
    })
}

/// Ollama's native `/api/chat` reply: arguments as an object, no id.
fn ollama_tool_turn(name: &str, arguments: Value) -> Value {
    json!({
        "model": "qwen3:8b",
        "created_at": "2026-11-25T10:00:00Z",
        "message": {"role": "assistant", "content": "", "thinking": "The user wants light.",
                    "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 1200,
        "eval_count": 30
    })
}

fn ollama_final(text: &str) -> Value {
    json!({
        "model": "qwen3:8b",
        "created_at": "2026-11-25T10:00:05Z",
        "message": {"role": "assistant", "content": text},
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 1300,
        "eval_count": 12
    })
}

async fn run_command(h: &Harness, delivery: &str, trigger: &str) {
    let (ct, body) = multipart(Some(TONE), None, false);
    let (status, text) = send(
        h,
        signed_request(delivery, trigger, false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
    tick(h).await;
}

fn meta(h: &Harness) -> Vec<Value> {
    h.llm_meta.lock().unwrap().clone()
}

async fn job_provider(h: &Harness) -> Value {
    let job = h.db.get_job(1).await.unwrap().unwrap();
    serde_json::to_value(job.agent_state.unwrap()).unwrap()["provider"].clone()
}

#[tokio::test]
async fn openai_tool_loop_request_shape() {
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"openai\"\nmodel = \"gpt-6.2-sol\"\neffort = \"low\"\nmax_tokens = 2000\nmax_retries = 0\n",
        ),
        vec![
            chat_tool_turn(
                "call_abc",
                "home__set_light",
                r#"{"name":"kitchen","on":true}"#,
            ),
            chat_final("Kitchen lights on."),
        ],
        Some("Turn on the kitchen lights."),
        |home| vec![home],
    )
    .await;
    run_command(&h, "oa-1", "double-click-hold").await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert_eq!(notes[0].body, "Kitchen lights on.");

    let m = meta(&h);
    assert_eq!(m.len(), 2);
    assert_eq!(m[0]["path"], "/v1/chat/completions");
    assert_eq!(m[0]["headers"]["authorization"], "Bearer sk-openai-test");
    assert!(m[0]["headers"].get("x-api-key").is_none());

    let seen = h.claude_seen.lock().unwrap().clone();
    let first = &seen[0];
    assert_eq!(first["model"], "gpt-6.2-sol");
    assert_eq!(first["reasoning_effort"], "low");
    assert_eq!(first["max_completion_tokens"], 2000);
    assert!(first.get("max_tokens").is_none());
    for anthropic_only in ["system", "output_config", "cache_control", "fallbacks"] {
        assert!(first.get(anthropic_only).is_none(), "{anthropic_only}");
    }
    assert_eq!(first["messages"][0]["role"], "developer");
    assert!(
        first["messages"][0]["content"]
            .as_str()
            .unwrap()
            .ends_with("Command mode.")
    );
    assert_eq!(first["messages"][1]["role"], "user");
    assert!(
        first["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("Turn on the kitchen lights.")
    );
    let tools: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            assert_eq!(t["type"], "function");
            assert_eq!(t["function"]["parameters"]["type"], "object");
            t["function"]["name"].as_str().unwrap()
        })
        .collect();
    assert_eq!(tools, ["home__set_light", "home__unlock_door"]);

    let second = &seen[1];
    assert_eq!(second["tools"], first["tools"]);
    assert_eq!(second["messages"][0], first["messages"][0]);
    assert_eq!(second["messages"][1], first["messages"][1]);
    // The assistant turn is echoed with its tool call; response-only fields
    // (refusal) are not.
    assert_eq!(
        second["messages"][2],
        json!({"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc", "type": "function",
            "function": {"name": "home__set_light", "arguments": r#"{"name":"kitchen","on":true}"#}}]})
    );
    assert_eq!(
        second["messages"][3],
        json!({"role": "tool", "tool_call_id": "call_abc", "content": "kitchen is now on"})
    );

    assert_eq!(job_provider(&h).await, "openai");
    let kinds: Vec<String> =
        h.db.audit_entries(1)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
    assert!(kinds.iter().any(|k| k == "llm_response"), "{kinds:?}");
}

#[tokio::test]
async fn mistral_confirm_approve_round_trip() {
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"mistral\"\nmodel = \"mistral-small-latest\"\nmax_retries = 0\n",
        ),
        vec![
            chat_tool_turn("D681PevKs", "home__unlock_door", r#"{"door":"front"}"#),
            chat_final("Front door unlocked."),
        ],
        Some("unlock the front door"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "mi-1", "double-click-hold").await;
    assert!(
        h.mcp_calls.lock().unwrap().is_empty(),
        "must not run before approval"
    );
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    let notes = h.notes.take();
    let Kind::Confirm {
        approve_url: Some(approve),
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
    let (status, _) = send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    tick(&h).await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door front".to_string()]
    );
    assert_eq!(h.notes.take().last().unwrap().body, "Front door unlocked.");
    let m = meta(&h);
    assert_eq!(m[0]["path"], "/v1/chat/completions");
    assert_eq!(m[0]["headers"]["authorization"], "Bearer mistral-test");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["model"], "mistral-small-latest");
    assert_eq!(seen[0]["messages"][0]["role"], "system");
    assert!(
        seen[0].get("reasoning_effort").is_none(),
        "no effort configured"
    );
    assert!(seen[0].get("max_tokens").is_none(), "no limit configured");
    assert_eq!(seen[1]["messages"][2]["tool_calls"][0]["id"], "D681PevKs");
    assert_eq!(
        seen[1]["messages"][3],
        json!({"role": "tool", "tool_call_id": "D681PevKs", "name": "home__unlock_door", "content": "unlocked"})
    );
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::Done
    );
}

#[tokio::test]
async fn openrouter_sends_attribution_and_echoes_reasoning() {
    let mut tool_turn = chat_tool_turn(
        "toolu_or1",
        "home__set_light",
        r#"{"name":"hall","on":false}"#,
    );
    let details = json!([{"type": "reasoning.encrypted", "data": "opaque", "id": "r1",
                          "format": "anthropic-claude-v1", "index": 0}]);
    tool_turn["choices"][0]["message"]["reasoning"] = json!("Turn off the hall light.");
    tool_turn["choices"][0]["message"]["reasoning_details"] = details.clone();
    tool_turn["choices"][0]["native_finish_reason"] = json!("tool_use");
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"openrouter\"\nmodel = \"anthropic/claude-sonnet-5.5\"\neffort = \"medium\"\nmax_retries = 0\n\n[llm.openrouter]\napp_url = \"https://github.com/lazzurs/ringmaster\"\napp_title = \"Ringmaster\"\n",
        ),
        vec![tool_turn, chat_final("Hall light off.")],
        Some("hall light off"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "or-1", "double-click-hold").await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light hall false".to_string()]
    );
    assert_eq!(h.notes.take()[0].body, "Hall light off.");
    let m = meta(&h);
    assert_eq!(m[0]["headers"]["authorization"], "Bearer sk-or-test");
    assert_eq!(
        m[0]["headers"]["http-referer"],
        "https://github.com/lazzurs/ringmaster"
    );
    assert_eq!(m[0]["headers"]["x-openrouter-title"], "Ringmaster");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen[0]["reasoning"], json!({"effort": "medium"}));
    assert!(seen[0].get("reasoning_effort").is_none());
    let echoed = &seen[1]["messages"][2];
    assert_eq!(
        echoed["reasoning_details"], details,
        "passed back unchanged"
    );
    assert!(echoed.get("reasoning").is_none());
    assert_eq!(seen[1]["messages"][3]["tool_call_id"], "toolu_or1");
}

#[tokio::test]
async fn ollama_native_tool_loop_without_auth() {
    let h = harness_config(
        &config_with(
            "[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\neffort = \"none\"\nmax_retries = 0\n",
        ),
        vec![
            ollama_tool_turn("home__set_light", json!({"name": "kitchen", "on": true})),
            ollama_final("Kitchen light is on."),
        ],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "ol-1", "double-click-hold").await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    assert_eq!(h.notes.take()[0].body, "Kitchen light is on.");

    let m = meta(&h);
    assert_eq!(m[0]["path"], "/api/chat");
    assert!(
        m[0]["headers"].get("authorization").is_none(),
        "Ollama needs no key"
    );
    let seen = h.claude_seen.lock().unwrap().clone();
    let first = &seen[0];
    assert_eq!(first["model"], "qwen3:8b");
    assert_eq!(first["stream"], false);
    assert_eq!(first["truncate"], false);
    assert_eq!(first["options"]["num_ctx"], 8192);
    assert_eq!(first["think"], false);
    assert_eq!(first["messages"][0]["role"], "system");
    assert_eq!(first["tools"][0]["type"], "function");

    // Object arguments, a generated id and the thinking are echoed back.
    let echoed = &seen[1]["messages"][2];
    let id = echoed["tool_calls"][0]["id"].as_str().unwrap();
    assert_eq!(id.len(), 9);
    assert_eq!(
        echoed["tool_calls"][0]["function"]["arguments"],
        json!({"name": "kitchen", "on": true})
    );
    assert_eq!(echoed["thinking"], "The user wants light.");
    assert_eq!(
        seen[1]["messages"][3],
        json!({"role": "tool", "tool_call_id": id, "tool_name": "home__set_light", "content": "kitchen is now on"})
    );
    assert_eq!(job_provider(&h).await, "ollama");
}

#[tokio::test]
async fn ollama_model_without_tools_runs_without_them() {
    let h = harness_config(
        &config_with("[llm]\nprovider = \"ollama\"\nmodel = \"gemma3:4b\"\nmax_retries = 0\n"),
        vec![
            json!({"__status": 400, "__body": {"error": "registry.ollama.ai/library/gemma3:4b does not support tools"}}),
            ollama_final("I can't control lights with this model."),
        ],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "ol-2", "double-click-hold").await;
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "retried once");
    assert!(seen[0].get("tools").is_some());
    assert!(seen[1].get("tools").is_none());
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert!(
        notes[0].body.contains("Unavailable: all tools"),
        "{}",
        notes[0].body
    );
    assert!(h.mcp_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_tool_arguments_are_never_run() {
    let h = harness_config(
        &config_with("[llm]\nprovider = \"openai\"\nmodel = \"gpt-5.4-mini\"\nmax_retries = 0\n"),
        vec![
            // Truncated JSON from a weak model.
            chat_tool_turn(
                "call_bad",
                "home__set_light",
                r#"{"name": "kitchen", "on": tr"#,
            ),
            chat_final("Sorry, that went wrong."),
        ],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "bad-1", "double-click-hold").await;
    assert!(h.mcp_calls.lock().unwrap().is_empty(), "never run");
    let seen = h.claude_seen.lock().unwrap().clone();
    let result = &seen[1]["messages"][3];
    assert_eq!(result["role"], "tool");
    assert_eq!(result["tool_call_id"], "call_bad");
    let text = result["content"].as_str().unwrap();
    assert!(
        text.starts_with("Error: Not run: the arguments are not valid JSON"),
        "{text}"
    );
    assert_eq!(h.notes.take()[0].body, "Sorry, that went wrong.");
}

#[tokio::test]
async fn ollama_unparseable_tool_call_fails_cleanly() {
    let h = harness_config(
        &config_with("[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\nmax_retries = 0\n"),
        vec![json!({"__status": 500, "__body": {"error": "failed to parse JSON: unexpected end of JSON input"}})],
        Some("kitchen light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "ol-3", "double-click-hold").await;
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Failed, "not retried for an hour");
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].kind, Kind::Error);
    assert!(
        notes[0].body.contains("malformed tool call"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn profiles_can_use_different_providers() {
    // Default (legacy [claude]) for command; capture on a local model.
    let config = CONFIG.replace(
        "system_prompt = \"Capture mode.\"",
        "system_prompt = \"Capture mode.\"\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"",
    );
    let h = harness_config(
        &config,
        vec![
            ollama_final("Added: buy milk"),
            final_turn("Kitchen lights on."),
        ],
        Some("buy milk"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "mix-1", "single-click-hold").await;
    run_command(&h, "mix-2", "double-click-hold").await;
    let m = meta(&h);
    assert_eq!(m[0]["path"], "/api/chat");
    assert_eq!(m[1]["path"], "/v1/messages");
    assert_eq!(m[1]["headers"]["x-api-key"], "sk-test");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(seen[0]["model"], "qwen3:8b");
    assert_eq!(seen[1]["model"], "claude-opus-5-5");
    let notes = h.notes.take();
    assert_eq!(notes[0].body, "Added: buy milk");
    assert_eq!(notes[1].body, "Kitchen lights on.");
}

#[tokio::test]
async fn state_saved_before_providers_still_resumes() {
    // A job paused for confirmation by v0.1.2 has no `provider` in its state.
    let h = harness(
        vec![
            tool_use_turn("toolu_old", "home__unlock_door", json!({"door": "side"})),
            final_turn("Side door unlocked."),
        ],
        Some("unlock the side door"),
    )
    .await;
    run_command(&h, "old-1", "double-click-hold").await;
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    {
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        let n = conn
            .execute(
                "UPDATE jobs SET agent_state = json_remove(agent_state, '$.provider') WHERE id = 1",
                [],
            )
            .unwrap();
        assert_eq!(n, 1);
        let raw: String = conn
            .query_row("SELECT agent_state FROM jobs WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(!raw.contains("\"provider\""), "{raw}");
    }
    let notes = h.notes.take();
    let Kind::Confirm {
        approve_url: Some(approve),
        ..
    } = &notes[0].kind
    else {
        panic!()
    };
    let approve_path = approve
        .strip_prefix("https://ring.test")
        .unwrap()
        .to_string();
    send(
        &h,
        Request::post(&approve_path).body(Body::empty()).unwrap(),
    )
    .await;
    tick(&h).await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door side".to_string()]
    );
    assert_eq!(h.notes.take().last().unwrap().body, "Side door unlocked.");
    let m = meta(&h);
    assert_eq!(m[1]["path"], "/v1/messages");
    let seen = h.claude_seen.lock().unwrap().clone();
    assert_eq!(
        seen[1]["messages"][2]["content"][0]["tool_use_id"],
        "toolu_old"
    );
    assert_eq!(job_provider(&h).await, "anthropic");
}

#[tokio::test]
async fn provider_errors_echoing_the_key_are_redacted_everywhere() {
    // A provider (or a proxy in front of it) that quotes the request's
    // credentials in its error body.
    for (config, key) in [
        (CONFIG.to_string(), "sk-test"),
        (
            config_with("[llm]\nprovider = \"openai\"\nmodel = \"gpt-6.2-sol\"\nmax_retries = 0\n"),
            "sk-openai-test",
        ),
    ] {
        let h = harness_config(
            &config,
            vec![json!({"__status": 401, "__body": {"error": {
                "type": "authentication_error",
                "message": format!("invalid credentials: x-api-key={key}, Authorization: Bearer {key}"),
            }}})],
            Some("Turn on the kitchen lights."),
            |home| vec![home],
        )
        .await;
        run_command(&h, "redact-1", "double-click-hold").await;

        let notes = h.notes.take();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].kind, Kind::Error);
        assert!(notes[0].body.contains("[redacted]"), "{}", notes[0].body);
        assert!(!format!("{notes:?}").contains(key), "{notes:?}");
        let audit = format!("{:?}", h.db.audit_entries(1).await.unwrap());
        assert!(audit.contains("[redacted]"), "{audit}");
        assert!(!audit.contains(key), "{audit}");
        let last_error: Option<String> = rusqlite::Connection::open(&h.db_path)
            .unwrap()
            .query_row("SELECT last_error FROM jobs WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert!(!last_error.unwrap_or_default().contains(key));
    }
}

#[tokio::test]
async fn missing_key_for_unused_provider_is_fine() {
    use ringmaster::config::Secrets;
    let config = Config::from_toml(&config_with(
        "[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\n",
    ))
    .unwrap();
    let secrets = Secrets {
        webhook_secret: Some(SECRET.into()),
        ..Default::default()
    };
    config.validate_runtime(&secrets).unwrap();

    // A profile on OpenAI makes its key required.
    let config = Config::from_toml(&CONFIG.replace(
        "system_prompt = \"Capture mode.\"",
        "system_prompt = \"Capture mode.\"\nprovider = \"openai\"\nmodel = \"gpt-5.4-mini\"",
    ))
    .unwrap();
    let mut secrets = Secrets {
        webhook_secret: Some(SECRET.into()),
        ..Default::default()
    };
    secrets
        .llm_keys
        .insert(Provider::Anthropic, "sk-ant".into());
    let err = config.validate_runtime(&secrets).unwrap_err().to_string();
    assert!(err.contains("OPENAI_API_KEY"), "{err}");
    secrets.llm_keys.insert(Provider::OpenAi, "sk".into());
    config.validate_runtime(&secrets).unwrap();
}

// ------------------------------------------------- one job never blocks the rest

/// The test config with extra `[agent]` keys.
fn config_agent(extra: &str) -> String {
    let anchor = "tool_discovery_timeout_secs = 2\n";
    assert!(CONFIG.contains(anchor));
    CONFIG.replace(anchor, &format!("{anchor}{extra}\n"))
}

async fn queue_command(h: &Harness, delivery: &str, phone_text: &str) {
    let (ct, body) = multipart(Some(TONE), Some(phone_text), false);
    let (status, text) = send(
        h,
        signed_request(delivery, "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
}

fn job_row(h: &Harness, id: i64) -> (String, Option<String>) {
    let conn = rusqlite::Connection::open(&h.db_path).unwrap();
    conn.query_row(
        "SELECT status, last_error FROM jobs WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
    .unwrap()
}

/// Model requests whose body contains `marker` never get an answer.
async fn hang_requests_containing(h: &Harness, marker: &str) {
    Mock::given(method("POST"))
        .and(wiremock::matchers::body_string_contains(marker))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(600)))
        .with_priority(1)
        .mount(&h._servers[1])
        .await;
}

#[tokio::test]
async fn unreadable_job_at_the_head_is_quarantined_and_the_queue_moves_on() {
    // Whisper is down, so each job's transcript is its phone transcription.
    let h = harness(vec![final_turn("Hall light on.")], None).await;
    queue_command(&h, "bad-1", "add milk to the shopping list").await;
    queue_command(&h, "good-2", "turn on the hall light").await;
    {
        // A state no version of Ringmaster can read, at the head of the queue.
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        conn.execute(
            "UPDATE jobs SET agent_state = '{\"messages\": 42, garbage' WHERE id = 1",
            [],
        )
        .unwrap();
    }

    tick(&h).await;

    let good = h.db.get_job(2).await.unwrap().unwrap();
    assert_eq!(good.status, JobStatus::Done);
    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "failed");
    let error = error.unwrap();
    assert!(error.contains("quarantined"), "{error}");
    assert!(error.contains("decoding agent state"), "{error}");

    let notes = h.notes.take();
    assert_eq!(notes.len(), 2, "{notes:?}");
    let failed = notes
        .iter()
        .find(|n| n.title == "Ringmaster: failed")
        .expect("the quarantined job is reported");
    assert_eq!(failed.kind, Kind::Error);
    assert!(failed.body.contains("could not be read"), "{}", failed.body);
    // The request is not lost: what the user said is in the notification.
    assert!(
        failed.body.contains("add milk to the shopping list"),
        "{}",
        failed.body
    );
    assert!(notes.iter().any(|n| n.body == "Hall light on."));

    let kinds: Vec<String> =
        h.db.audit_entries(1)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
    assert!(kinds.iter().any(|k| k == "job_quarantined"), "{kinds:?}");

    // Never picked up again, even after a restart's recovery sweep.
    h.db.requeue_running(now(), &[]).await.unwrap();
    tick(&h).await;
    tick(&h).await;
    assert!(h.notes.take().is_empty());
    assert_eq!(h.claude_seen.lock().unwrap().len(), 1);
    assert_eq!(job_row(&h, 1).0, "failed");
}

async fn hung_job_scenario(concurrency: usize) {
    let h = harness_config(
        &config_agent(&format!(
            "attempt_timeout_secs = 2\nworker_concurrency = {concurrency}"
        )),
        vec![final_turn("Kitchen lights on.")],
        None,
        |home| vec![home],
    )
    .await;
    hang_requests_containing(&h, "please hang").await;
    queue_command(&h, "hang-1", "please hang forever").await;
    queue_command(&h, "ok-2", "turn on the kitchen lights").await;

    // The old worker waited for the hung request (120 s) before job 2.
    tokio::time::timeout(Duration::from_secs(30), tick(&h))
        .await
        .expect("a hung job must not stall the worker");

    assert_eq!(
        h.db.get_job(2).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    // The hung attempt hit its deadline and is retried later, like an outage.
    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "queued");
    assert!(
        error.as_deref().unwrap().contains("attempt_timeout_secs"),
        "{error:?}"
    );
    let notes = h.notes.take();
    assert!(
        notes.iter().any(|n| n.body == "Kitchen lights on."),
        "{notes:?}"
    );
    assert!(
        notes
            .iter()
            .any(|n| n.title == "Ringmaster: delayed" && n.body.contains("keep trying")),
        "{notes:?}"
    );
    let audit: Vec<(String, Value)> = h.db.audit_entries(1).await.unwrap();
    assert!(
        audit.iter().any(|(k, d)| k == "retry_scheduled"
            && d["error"].as_str().unwrap().contains("no result within 2s")),
        "{audit:?}"
    );
}

#[tokio::test]
async fn hung_job_does_not_stop_others_with_concurrency() {
    hung_job_scenario(2).await;
}

#[tokio::test]
async fn hung_job_does_not_stop_others_one_at_a_time() {
    hung_job_scenario(1).await;
}

#[tokio::test]
async fn slow_job_does_not_delay_the_next_one() {
    // The deadline is far away: only concurrency lets job 2 through.
    let h = harness_config(
        &config_agent("attempt_timeout_secs = 300\nworker_concurrency = 2"),
        vec![final_turn("Kitchen lights on.")],
        None,
        |home| vec![home],
    )
    .await;
    hang_requests_containing(&h, "slow request").await;
    queue_command(&h, "slow-1", "a slow request").await;
    queue_command(&h, "fast-2", "turn on the kitchen lights").await;

    let stop = CancellationToken::new();
    let worker = tokio::spawn(h.worker.clone().run(stop.clone()));
    let done = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if h.db.get_job(2).await.unwrap().unwrap().status == JobStatus::Done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    // Still working on job 1 meanwhile.
    let slow = h.db.get_job(1).await.unwrap().unwrap().status;
    worker.abort();
    done.expect("job 2 finishes while job 1 is still running");
    assert_eq!(slow, JobStatus::Running);
}

/// Panics when a tool is called.
struct PanickingTools(Arc<dyn ToolBackend>);

#[async_trait]
impl ToolBackend for PanickingTools {
    async fn list_tools(&self, servers: &[String]) -> Result<Vec<DiscoveredTool>, ToolError> {
        self.0.list_tools(servers).await
    }
    async fn call_tool(
        &self,
        _server: &str,
        tool: &str,
        _args: serde_json::Map<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        panic!("injected panic in {tool}");
    }
    fn server_names(&self) -> Vec<String> {
        self.0.server_names()
    }
}

#[tokio::test]
async fn panic_in_one_job_fails_only_that_job() {
    let h = harness_full(
        &config_agent("worker_concurrency = 1"),
        vec![
            tool_use_turn(
                "toolu_p",
                "home__set_light",
                json!({"name": "porch", "on": true}),
            ),
            final_turn("Noted."),
        ],
        None,
        |home| vec![home],
        |tools| Arc::new(PanickingTools(tools)),
    )
    .await;
    queue_command(&h, "boom-1", "turn on the porch light").await;
    queue_command(&h, "ok-2", "remember to buy milk").await;

    tick(&h).await;

    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "failed");
    assert!(
        error
            .as_deref()
            .unwrap()
            .contains("injected panic in set_light"),
        "{error:?}"
    );
    assert_eq!(
        h.db.get_job(2).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 2, "{notes:?}");
    assert!(
        notes[0].body.contains("turn on the porch light"),
        "{notes:?}"
    );
    assert_eq!(notes[1].body, "Noted.");
    // Not retried, and not left claimed.
    tick(&h).await;
    assert_eq!(h.claude_seen.lock().unwrap().len(), 2);
    assert_eq!(job_row(&h, 1).0, "failed");
}

#[tokio::test]
async fn job_awaiting_confirmation_does_not_block_others() {
    let h = harness_config(
        &config_agent("worker_concurrency = 1"),
        vec![
            tool_use_turn("toolu_c", "home__unlock_door", json!({"door": "front"})),
            final_turn("Kitchen lights on."),
            final_turn("It is noon."),
        ],
        None,
        |home| vec![home],
    )
    .await;
    queue_command(&h, "confirm-1", "unlock the front door").await;
    queue_command(&h, "after-2", "turn on the kitchen lights").await;

    tick(&h).await;

    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    assert_eq!(
        h.db.get_job(2).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    let notes = h.notes.take();
    assert!(matches!(notes[0].kind, Kind::Confirm { .. }), "{notes:?}");
    assert_eq!(notes[1].body, "Kitchen lights on.");

    // A third request is not held up by the pending one either, and the
    // pending job is not claimed while it waits.
    queue_command(&h, "later-3", "what time is it").await;
    tick(&h).await;
    assert_eq!(
        h.db.get_job(3).await.unwrap().unwrap().status,
        JobStatus::Done
    );
    assert_eq!(h.notes.take()[0].body, "It is noon.");
    assert_eq!(h.claude_seen.lock().unwrap().len(), 3);
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    assert!(h.mcp_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn started_job_is_not_resumed_long_after_its_max_age() {
    let h = harness(
        vec![
            json!({"__status": 503, "__body": {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}}),
            final_turn("Heating off."),
        ],
        None,
    )
    .await;
    queue_command(&h, "late-1", "turn the heating off").await;
    tick(&h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Queued);
    assert!(job.agent_state.is_some(), "the conversation was started");
    h.notes.take();

    // The service was down (or kept crashing on this job) for a day; the
    // job comes back through the restart sweep, not a scheduled retry.
    {
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        let day_ago = now() - 86_400;
        conn.execute(
            "UPDATE jobs SET status = 'running', created_at = ?1 WHERE id = 1",
            [day_ago],
        )
        .unwrap();
        conn.execute(
            "UPDATE deliveries SET recorded_at_ms = ?1",
            [day_ago * 1000],
        )
        .unwrap();
    }
    tick(&h).await;

    let (status, error) = job_row(&h, 1);
    assert_eq!(status, "failed");
    assert!(
        error.as_deref().unwrap().contains("too old to resume"),
        "{error:?}"
    );
    assert_eq!(h.claude_seen.lock().unwrap().len(), 1, "not resumed");
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    assert!(notes[0].body.contains("turn the heating off"), "{notes:?}");
}

// ------------------------------------------------- health, alerts, shutdown

async fn healthz(h: &Harness) -> (StatusCode, String) {
    send(h, Request::get("/healthz").body(Body::empty()).unwrap()).await
}

/// Alert and recovery notifications, leaving the others.
fn alerts(h: &Harness) -> Vec<Notification> {
    h.notes
        .take()
        .into_iter()
        .filter(|n| matches!(n.kind, Kind::Alert { .. } | Kind::Recovered))
        .collect()
}

/// The gauges as `/metrics` would render them right now.
fn gauges(h: &Harness) -> String {
    let rec = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = rec.handle();
    metrics::with_local_recorder(&rec, || h.health.export(now(), &h.db.status().snapshot()));
    handle.render()
}

#[tokio::test]
async fn corrupt_database_alerts_once_refuses_webhooks_and_recovers_once() {
    let h = harness(vec![final_turn("Done.")], None).await;
    queue_command(&h, "before-1", "turn on the kitchen lights").await;
    h.watchdog.check(now()).await;
    assert!(alerts(&h).is_empty());
    assert_eq!(healthz(&h).await.0, StatusCode::OK);
    assert!(gauges(&h).contains("ringmaster_db_healthy 1"));

    // Damage the file on disk (its header), as a failing disk would.
    rusqlite::Connection::open(&h.db_path)
        .unwrap()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    let good = std::fs::read(&h.db_path).unwrap();
    let mut bad = good.clone();
    bad[..100].fill(0xA5);
    std::fs::write(&h.db_path, &bad).unwrap();

    h.watchdog.check(now()).await;
    h.watchdog.check(now()).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1, "one alert, debounced: {sent:?}");
    assert_eq!(sent[0].title, "Ringmaster alert: database corrupt");
    assert_eq!(sent[0].kind, Kind::Alert { reminder: false });
    assert!(sent[0].body.contains("SQLITE_NOTADB"), "{}", sent[0].body);
    assert!(
        sent[0].body.contains("may not be stored or processed"),
        "{}",
        sent[0].body
    );
    let (status, body) = healthz(&h).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("database corrupt"), "{body}");
    let g = gauges(&h);
    assert!(g.contains("ringmaster_db_healthy 0"), "{g}");
    assert!(g.contains("ringmaster_alerts_active 1"), "{g}");
    assert!(
        g.contains("ringmaster_alert_active{condition=\"database_corrupt\"} 1"),
        "{g}"
    );

    // The phone is told to retry rather than given a 2xx for data that is
    // not safely stored.
    let (ct, body) = multipart(Some(TONE), Some("add milk"), false);
    let req = signed_request("during-2", "double-click-hold", false, &ct, body, now());
    let (status, text) = send(&h, req).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{text}");

    // Restored from a backup.
    std::fs::write(&h.db_path, &good).unwrap();
    h.watchdog.check(now()).await;
    h.watchdog.check(now()).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1, "one recovery message: {sent:?}");
    assert_eq!(sent[0].title, "Ringmaster recovered: database corrupt");
    assert_eq!(sent[0].kind, Kind::Recovered);
    assert_eq!(healthz(&h).await.0, StatusCode::OK);
    assert!(gauges(&h).contains("ringmaster_db_healthy 1"));
    let (ct, body) = multipart(Some(TONE), Some("add milk"), false);
    let req = signed_request("during-2", "double-click-hold", false, &ct, body, now());
    assert_eq!(send(&h, req).await.0, StatusCode::ACCEPTED);
}

#[tokio::test]
async fn stalled_worker_and_queue_alert_once_and_recover() {
    let h = harness(vec![final_turn("Kitchen lights on.")], None).await;
    // The worker loop stops beating (stuck), with a job waiting.
    queue_command(&h, "wait-1", "turn on the kitchen lights").await;
    let t = now();
    h.health.beat_worker(t - 300);
    {
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        conn.execute("UPDATE jobs SET next_attempt_at = ?1", [t - 1000])
            .unwrap();
    }
    for _ in 0..3 {
        h.watchdog.check(t).await;
    }
    let sent = alerts(&h);
    let titles: Vec<&str> = sent.iter().map(|n| n.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Ringmaster alert: worker stalled",
            "Ringmaster alert: queue stalled"
        ],
        "{sent:?}"
    );
    assert!(
        sent[0].body.contains("has not run for 5 min"),
        "{}",
        sent[0].body
    );
    assert!(
        sent[1].body.starts_with("Job 1 has been due for 16 min"),
        "{}",
        sent[1].body
    );
    let (status, body) = healthz(&h).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("worker loop stalled"), "{body}");
    assert!(gauges(&h).contains("ringmaster_oldest_due_job_age_seconds 1000"));

    // A reminder after an hour, not before.
    h.watchdog.check(t + 3599).await;
    assert!(alerts(&h).is_empty());
    h.watchdog.check(t + 3600).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 2, "{sent:?}");
    assert_eq!(sent[1].title, "Ringmaster alert (ongoing): queue stalled");
    assert_eq!(sent[1].kind, Kind::Alert { reminder: true });
    assert!(
        sent[1].body.ends_with("Ongoing for 60 min."),
        "{}",
        sent[1].body
    );

    // The loop runs again and the job goes through: one recovery each.
    tick(&h).await;
    let t = now();
    h.health.beat_worker(t);
    h.watchdog.check(t).await;
    h.watchdog.check(t).await;
    let sent = alerts(&h);
    let titles: Vec<&str> = sent.iter().map(|n| n.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Ringmaster recovered: worker stalled",
            "Ringmaster recovered: queue stalled"
        ]
    );
    assert!(sent.iter().all(|n| n.kind == Kind::Recovered));
    assert_eq!(healthz(&h).await.0, StatusCode::OK);
}

#[tokio::test]
async fn job_past_its_deadline_alerts_and_recovers() {
    let h = harness(vec![], None).await;
    // A job whose attempt deadline did not fire (e.g. blocking code that
    // never yields): its task has run far past attempt_timeout_secs.
    let t = now();
    h.health.job_started(42, t - 2000);
    h.watchdog.check(t).await;
    h.watchdog.check(t + 30).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0].title, "Ringmaster alert: job past its deadline");
    assert!(
        sent[0].body.starts_with("Job 42 running for up to 33 min"),
        "{}",
        sent[0].body
    );
    // Not a reason to restart the container.
    assert_eq!(healthz(&h).await.0, StatusCode::OK);
    h.health.job_finished(42);
    h.watchdog.check(t + 60).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].title, "Ringmaster recovered: job past its deadline");
}

#[tokio::test]
async fn alerts_never_carry_transcripts_or_secrets() {
    let h = harness(vec![], Some("unlock the garden shed")).await;
    for (i, text) in ["the pin is 4321", "tell Anna her code", "open sesame"]
        .iter()
        .enumerate()
    {
        queue_command(&h, &format!("bad-{i}"), text).await;
    }
    {
        let conn = rusqlite::Connection::open(&h.db_path).unwrap();
        conn.execute(
            "UPDATE jobs SET agent_state = '{broken', transcript = 'unlock the garden shed'",
            [],
        )
        .unwrap();
    }
    tick(&h).await; // three quarantined jobs
    // The database file turns read-only: the write probe fails.
    set_read_only(&h.db_path, true);
    h.health.beat_worker(now() - 600);
    h.watchdog.check(now()).await;
    let sent = alerts(&h);
    let kinds: Vec<&str> = sent.iter().map(|n| n.title.as_str()).collect();
    assert!(
        kinds.contains(&"Ringmaster alert: jobs failing"),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&"Ringmaster alert: database failing"),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&"Ringmaster alert: worker stalled"),
        "{kinds:?}"
    );
    let failing = sent
        .iter()
        .find(|n| n.title.ends_with("jobs failing"))
        .unwrap();
    assert!(failing.body.contains("quarantined 3"), "{}", failing.body);
    for n in &sent {
        let text = format!("{} {}", n.title, n.body);
        for secret in [
            "unlock the garden shed",
            "4321",
            "Anna",
            "sesame",
            SECRET,
            "sk-test",
            "sk-openai-test",
        ] {
            assert!(!text.contains(secret), "{secret:?} leaked into {text}");
        }
    }
    set_read_only(&h.db_path, false);
}

/// The database file and its write-ahead log read-only (or writable again),
/// as after a read-only remount.
fn set_read_only(path: &std::path::Path, ro: bool) {
    let wal = path.with_extension("db-wal");
    for p in [path, wal.as_path()] {
        if let Ok(m) = std::fs::metadata(p) {
            let mut perms = m.permissions();
            perms.set_readonly(ro);
            std::fs::set_permissions(p, perms).unwrap();
        }
    }
}

#[tokio::test]
async fn read_only_database_alerts_and_recovers() {
    let h = harness(vec![], None).await;
    set_read_only(&h.db_path, true);
    h.watchdog.check(now()).await;
    h.watchdog.check(now()).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0].title, "Ringmaster alert: database failing");
    assert!(
        sent[0]
            .body
            .starts_with("Database writes fail: database is read-only (SQLITE_READONLY)."),
        "{}",
        sent[0].body
    );
    assert!(sent[0].body.contains("may not be stored or processed"));
    let (status, body) = healthz(&h).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("SQLITE_READONLY"), "{body}");
    assert!(gauges(&h).contains("ringmaster_db_healthy 0"));

    set_read_only(&h.db_path, false);
    h.watchdog.check(now()).await;
    h.watchdog.check(now()).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0].title, "Ringmaster recovered: database failing");
    assert_eq!(healthz(&h).await.0, StatusCode::OK);
}

#[tokio::test]
async fn stuck_decoder_never_holds_up_jobs_or_piles_up_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let (l, p) = (live.clone(), peak.clone());
    // A decoder that ignores its deadline, as a hang inside the codec would.
    let hang: ringmaster::audio::DecodeFn =
        Arc::new(move |_: &[u8], _: &ringmaster::audio::Budget| {
            let n = l.fetch_add(1, Ordering::SeqCst) + 1;
            p.fetch_max(n, Ordering::SeqCst);
            std::thread::sleep(Duration::from_secs(4));
            l.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("gave up")
        });
    let config = CONFIG.replace(
        "vocabulary = [\"kitchen\"]\n",
        "vocabulary = [\"kitchen\"]\ndecode_timeout_secs = 1\ndecode_concurrency = 1\n",
    ) + "\n[alerts]\noverrun_margin_secs = 0\n";
    let config = config.replace(
        "tool_discovery_timeout_secs = 2\n",
        "tool_discovery_timeout_secs = 2\nworker_concurrency = 3\n",
    );
    let h = harness_decoder(
        &config,
        vec![final_turn("One."), final_turn("Two."), final_turn("Three.")],
        Some("never used"),
        |home| vec![home],
        |t| t,
        Box::new(move |c: &Config, health: &Health| {
            Decoder::with_fn(&c.stt, health.blocking.clone(), hang)
        }),
    )
    .await;
    for i in 0..3 {
        queue_command(&h, &format!("rec-{i}"), "what time is it").await;
    }

    let started = std::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(4), tick(&h))
        .await
        .expect("a stuck decoder must not hold the jobs up");
    assert!(started.elapsed() < Duration::from_secs(4));
    for id in 1..=3 {
        let job = h.db.get_job(id).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Done);
        assert_eq!(job.transcript_source.as_deref(), Some("phone"));
    }
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "never more than decode_concurrency threads"
    );
    assert_eq!(h.health.blocking.count("decode"), 1);
    let audit = h.db.audit_entries(1).await.unwrap();
    assert!(
        audit.iter().any(|(k, _)| k == "audio_decode_failed"),
        "{audit:?}"
    );

    // The stuck thread is visible to the watchdog until it returns.
    h.notes.take();
    h.watchdog.check(now() + 3).await;
    let sent = alerts(&h);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0].title, "Ringmaster alert: blocking work stuck");
    assert!(sent[0].body.contains("1 decode"), "{}", sent[0].body);
    assert!(gauges(&h).contains("ringmaster_blocking_tasks{kind=\"decode\"} 1"));
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(h.health.blocking.count("decode"), 0);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn shutdown_ends_within_the_grace_period_and_requeues_the_running_job() {
    let h = harness_config(
        &config_agent("attempt_timeout_secs = 300"),
        vec![],
        None,
        |home| vec![home],
    )
    .await;
    hang_requests_containing(&h, "please hang").await;
    queue_command(&h, "hang-1", "please hang forever").await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stop = CancellationToken::new();
    let service = tokio::spawn(ringmaster::run_service(
        listener,
        h.app.clone(),
        h.worker.clone(),
        h.watchdog.clone(),
        stop.clone(),
        Duration::from_secs(2),
    ));
    tokio::time::timeout(Duration::from_secs(10), async {
        while h.db.get_job(1).await.unwrap().unwrap().status != JobStatus::Running {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the job starts");
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let url = format!("http://{addr}/healthz");
    assert_eq!(client.get(&url).send().await.unwrap().status(), 200);

    let started = std::time::Instant::now();
    stop.cancel();
    let report = tokio::time::timeout(Duration::from_secs(5), service)
        .await
        .expect("shutdown is bounded")
        .unwrap()
        .unwrap();
    let took = started.elapsed();
    assert!(
        took >= Duration::from_secs(2),
        "waited for the job: {took:?}"
    );
    assert!(took < Duration::from_millis(3500), "{took:?}");
    assert_eq!(report.jobs_left, vec![1]);

    // Nothing is listening any more, and the job is left for the next start.
    assert!(client.get(&url).send().await.is_err());
    assert_eq!(job_row(&h, 1).0, "running");
    assert_eq!(h.db.requeue_running(now(), &[]).await.unwrap(), 1);
    assert_eq!(job_row(&h, 1).0, "queued");
}

#[tokio::test]
async fn idle_mcp_sessions_are_closed_and_their_tasks_end() {
    let (url, _calls, _stop) = start_mcp().await;
    let backend = McpBackend::new(
        vec![McpServerConfig {
            name: "home".into(),
            url,
            ..Default::default()
        }],
        Default::default(),
        Duration::from_secs(5),
    );
    let alive = || {
        tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks()
    };
    // Tasks still alive a moment after the session is closed.
    async fn settle(alive: &impl Fn() -> usize) -> usize {
        let mut last = alive();
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let n = alive();
            if n == last && n < 5 {
                break;
            }
            last = n;
        }
        alive()
    }
    let mut after_close = Vec::new();
    for round in 0..3 {
        assert_eq!(backend.list_tools(&["home".into()]).await.unwrap().len(), 3);
        assert_eq!(backend.sessions().await, 1);
        let open = alive();
        // Recently used: kept.
        assert_eq!(
            backend.close_idle(Duration::from_secs(600)).await,
            0,
            "round {round}"
        );
        assert_eq!(backend.close_idle(Duration::ZERO).await, 1);
        assert_eq!(backend.sessions().await, 0);
        let left = settle(&alive).await;
        assert!(
            left + 5 <= open,
            "the session's tasks end: {left} of {open}"
        );
        after_close.push(left);
    }
    // Sessions opened and closed over and over leave nothing behind.
    assert!(
        after_close.windows(2).all(|w| w[1] <= w[0]),
        "{after_close:?}"
    );
}

// ------------------------------------------------- approval classifier

/// An assistant turn with several tool calls.
fn multi_tool_turn(calls: &[(&str, &str, Value)]) -> Value {
    let mut content = vec![json!({"type": "thinking", "thinking": "", "signature": "sig-multi"})];
    for (id, name, input) in calls {
        content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
    }
    json!({
        "id": "msg_multi",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "tool_use",
        "content": content,
        "usage": {"input_tokens": 900, "output_tokens": 40}
    })
}

/// A llama-server style reply whose yes/no logprobs give `score`.
fn shield_reply(score: f64) -> ResponseTemplate {
    let (yes, no) = (score.ln(), (1.0 - score).ln());
    let (first, lp) = if yes > no { ("yes", yes) } else { ("no", no) };
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{
            "index": 0,
            "finish_reason": "length",
            "message": {"role": "assistant", "content": first},
            "logprobs": {"content": [{
                "id": 1, "token": first, "bytes": [], "logprob": lp,
                "top_logprobs": [
                    {"id": 2, "token": "no", "bytes": [], "logprob": no},
                    {"id": 3, "token": "yes", "bytes": [], "logprob": yes},
                    {"id": 4, "token": "No", "bytes": [], "logprob": -12.0},
                    {"id": 5, "token": "The", "bytes": [], "logprob": -14.0}
                ]
            }]}
        }],
        "object": "chat.completion",
        "model": "shieldstral"
    }))
}

/// Answers each classifier request with `f(body)`.
struct Shield(Box<dyn Fn(&str) -> ResponseTemplate + Send + Sync>);

impl Respond for Shield {
    fn respond(&self, req: &MockRequest) -> ResponseTemplate {
        (self.0)(std::str::from_utf8(&req.body).unwrap())
    }
}

async fn shield_server(f: impl Fn(&str) -> ResponseTemplate + Send + Sync + 'static) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Shield(Box::new(f)))
        .mount(&server)
        .await;
    server
}

/// The test config with every `home/*` tool on `confirm` (unlock on
/// `unlock_policy`), deletes on `deny`, and `[approval]` in `mode` against
/// the classifier at `url`.
fn approval_config(url: &str, mode: &str, unlock_policy: &str, extra: &str) -> String {
    let policies = "[[tool_policies]]\nmatch = \"home/unlock_*\"\npolicy = \"confirm\"\n\n\
                    [[tool_policies]]\nmatch = \"home/*\"\npolicy = \"allow\"\n";
    assert!(CONFIG.contains(policies));
    let config = CONFIG.replace(
        policies,
        &format!(
            "[[tool_policies]]\nmatch = \"home/unlock_*\"\npolicy = \"{unlock_policy}\"\n\n\
             [[tool_policies]]\nmatch = \"home/*\"\npolicy = \"confirm\"\n"
        ),
    );
    format!(
        "{config}\n[approval]\nmode = \"{mode}\"\nbase_url = \"{url}\"\nauto_approve_below = 0.2\n\
         request_timeout_secs = 1\nmax_retries = 0\n{extra}\n"
    )
}

async fn audit_of(h: &Harness, kind: &str) -> Vec<Value> {
    h.db.audit_entries(1)
        .await
        .unwrap()
        .into_iter()
        .filter(|(k, _)| k == kind)
        .map(|(_, v)| v)
        .collect()
}

/// The path of a confirmation notification's Approve or Deny button.
fn button(n: &Notification, approve: bool) -> String {
    let Kind::Confirm {
        approve_url: Some(a),
        deny_url: Some(d),
        ..
    } = &n.kind
    else {
        panic!("expected a confirmation: {n:?}");
    };
    let url = if approve { a } else { d };
    url.strip_prefix("https://ring.test").unwrap().to_string()
}

#[tokio::test]
async fn classifier_low_score_runs_the_tool_and_says_so() {
    let shield = shield_server(|_| shield_reply(0.02)).await;
    let h = harness_config(
        &approval_config(&shield.uri(), "classifier", "confirm", ""),
        vec![
            multi_tool_turn(&[(
                "toolu_1",
                "home__set_light",
                json!({"name": "kitchen", "on": true}),
            )]),
            final_turn("Kitchen lights on."),
        ],
        Some(" Turn on the kitchen lights."),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-1", "double-click-hold").await;

    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["set_light kitchen true".to_string()]
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "no confirmation asked: {notes:?}");
    assert_eq!(notes[0].kind, Kind::Result);
    assert_eq!(
        notes[0].body,
        "Kitchen lights on.\n\nAuto-approved by the classifier: home/set_light \
         {\"name\":\"kitchen\",\"on\":true} (score 0.02)"
    );
    let c = &audit_of(&h, "classification").await[0];
    assert_eq!(c["verdict"], "safe");
    assert_eq!(c["outcome"], "auto_approved");
    assert_eq!(c["mode"], "classifier");
    assert!((c["score"].as_f64().unwrap() - 0.02).abs() < 1e-9, "{c}");
    assert_eq!(c["threshold"], 0.5);
    assert_eq!(c["auto_approve_below"], 0.2);
    let call = &audit_of(&h, "tool_call").await[0];
    assert_eq!(call["approved_by"], "classifier");
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::Done
    );
}

#[tokio::test]
async fn classifier_request_matches_the_model_card() {
    let shield = shield_server(|_| shield_reply(0.01)).await;
    let h = harness_config(
        &approval_config(
            &shield.uri(),
            "classifier",
            "confirm",
            "model = \"shieldstral-test\"",
        ),
        vec![
            multi_tool_turn(&[(
                "toolu_1",
                "home__set_light",
                json!({"name": "kitchen", "on": true}),
            )]),
            final_turn("Done."),
        ],
        Some("turn on the kitchen lights"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-2", "double-click-hold").await;

    let reqs = shield.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1);
    let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(body["model"], "shieldstral-test");
    assert_eq!(body["max_tokens"], 1);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["logprobs"], true);
    assert_eq!(body["top_logprobs"], 20);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(
        messages[0]["content"],
        "Judge whether the Document meets the requirements based on the Query and the \
         Instruction provided. Note that the answer can only be \"yes\" or \"no\"."
    );
    assert_eq!(messages[1]["role"], "user");
    let user = messages[1]["content"].as_str().unwrap();
    assert!(
        user.starts_with("<Instruct>: You are a strict safety reviewer"),
        "{user}"
    );
    assert!(
        user.contains("\n\n<Query>: Does this proposed action go beyond what the user asked for"),
        "{user}"
    );
    assert!(
        user.ends_with(
            "\n\n<Document>: [User request]\nturn on the kitchen lights\n\n\
             [Proposed action]\nTool: home/set_light\nDescription: Turn a light on or off\n\
             Arguments: {\"name\":\"kitchen\",\"on\":true}"
        ),
        "{user}"
    );
    // No credentials were configured, so none are sent.
    assert!(reqs[0].headers.get("authorization").is_none());
}

#[tokio::test]
async fn classifier_flag_escalates_and_never_denies() {
    let shield = shield_server(|_| shield_reply(0.93)).await;
    let h = harness_config(
        &approval_config(&shield.uri(), "classifier", "confirm", ""),
        vec![
            tool_use_turn("toolu_9", "home__unlock_door", json!({"door": "front"})),
            final_turn("Front door unlocked."),
        ],
        Some("unlock the front door"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-3", "double-click-hold").await;

    assert!(h.mcp_calls.lock().unwrap().is_empty());
    assert_eq!(
        h.db.get_job(1).await.unwrap().unwrap().status,
        JobStatus::AwaitingConfirmation
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(
        notes[0].title,
        "Ringmaster: approve? (flagged by the classifier)"
    );
    assert!(
        notes[0].body.contains(
            "home/unlock_door {\"door\":\"front\"}\n  Classifier flagged this (score 0.93 > 0.50)"
        ),
        "{}",
        notes[0].body
    );
    let c = &audit_of(&h, "classification").await[0];
    assert_eq!(
        (c["verdict"].as_str(), c["outcome"].as_str()),
        (Some("unsafe"), Some("escalated"))
    );

    // It did not deny anything: the human can still approve.
    let (status, _) = send(
        &h,
        Request::post(button(&notes[0], true))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    tick(&h).await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec!["unlock_door front".to_string()]
    );
    let last = h.notes.take().pop().unwrap();
    // Approved by a human: not listed as auto-approved.
    assert_eq!(last.body, "Front door unlocked.");
    assert_eq!(audit_of(&h, "tool_call").await[0]["approved_by"], "human");
}

#[tokio::test]
async fn classifier_grey_band_escalates() {
    let shield = shield_server(|_| shield_reply(0.35)).await;
    let h = harness_config(
        &approval_config(&shield.uri(), "classifier", "confirm", ""),
        vec![multi_tool_turn(&[(
            "toolu_1",
            "home__set_light",
            json!({"name": "hall", "on": false}),
        )])],
        Some("turn off the hall light"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-4", "double-click-hold").await;

    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let notes = h.notes.take();
    assert!(matches!(notes[0].kind, Kind::Confirm { .. }), "{notes:?}");
    assert_eq!(notes[0].title, "Ringmaster: approve?");
    assert!(
        notes[0]
            .body
            .contains("Classifier unsure (score 0.35; only below 0.20 runs without asking)"),
        "{}",
        notes[0].body
    );
    assert_eq!(
        audit_of(&h, "classification").await[0]["verdict"],
        "uncertain"
    );
}

#[tokio::test]
async fn classifier_down_slow_or_malformed_asks_the_human() {
    let dead = {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        format!("http://127.0.0.1:{port}")
    };
    let slow = shield_server(|_| shield_reply(0.01).set_delay(Duration::from_secs(5))).await;
    let malformed = shield_server(|_| {
        ResponseTemplate::new(200)
            .set_body_json(json!({"choices": [{"message": {"content": "no"}}]}))
    })
    .await;
    let not_found =
        shield_server(|_| ResponseTemplate::new(404).set_body_string("no such model")).await;
    for (url, expect) in [
        (dead, "classifier unavailable"),
        (slow.uri(), "no verdict within 1 s"),
        (malformed.uri(), "unusable classifier reply: no logprobs"),
        (not_found.uri(), "classifier rejected the request: 404"),
    ] {
        let h = harness_config(
            &approval_config(&url, "classifier", "confirm", "alert_after_failures = 2"),
            vec![
                multi_tool_turn(&[(
                    "toolu_1",
                    "home__set_light",
                    json!({"name": "kitchen", "on": true}),
                )]),
                multi_tool_turn(&[(
                    "toolu_2",
                    "home__set_light",
                    json!({"name": "hall", "on": true}),
                )]),
            ],
            Some("turn on the lights"),
            |home| vec![home],
        )
        .await;
        let started = std::time::Instant::now();
        run_command(&h, "rec-shield-5a", "double-click-hold").await;
        assert!(started.elapsed() < Duration::from_secs(4), "{url}");

        assert!(h.mcp_calls.lock().unwrap().is_empty(), "{url}");
        let job = h.db.get_job(1).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::AwaitingConfirmation, "{url}");
        let notes = h.notes.take();
        assert_eq!(notes.len(), 1, "{url}: {notes:?}");
        assert!(
            notes[0].body.contains("No classifier verdict (") && notes[0].body.contains(expect),
            "{url}: {}",
            notes[0].body
        );
        let c = &audit_of(&h, "classification").await[0];
        assert_eq!(c["verdict"], "unavailable", "{url}");
        assert!(c["score"].is_null());

        // A second failure in a row raises the operator alert.
        run_command(&h, "rec-shield-5b", "double-click-hold").await;
        h.notes.take();
        h.health.beat_worker(now());
        h.watchdog.check(now()).await;
        let sent = alerts(&h);
        assert_eq!(sent.len(), 1, "{url}: {sent:?}");
        assert_eq!(
            sent[0].title,
            "Ringmaster alert: approval classifier failing"
        );
        assert!(
            sent[0].body.contains("failed 2 times in a row"),
            "{}",
            sent[0].body
        );
        assert!(!sent[0].body.contains("lights"), "no transcripts in alerts");
    }
}

#[tokio::test]
async fn deny_and_ask_are_never_classified() {
    let shield = shield_server(|_| shield_reply(0.01)).await;
    let h = harness_config(
        &approval_config(&shield.uri(), "classifier", "ask", ""),
        vec![
            multi_tool_turn(&[
                ("toolu_d", "home__delete_everything", json!({})),
                ("toolu_u", "home__unlock_door", json!({"door": "back"})),
                (
                    "toolu_l",
                    "home__set_light",
                    json!({"name": "porch", "on": true}),
                ),
            ]),
            final_turn("Back door unlocked and porch light on."),
        ],
        Some("unlock the back door and turn on the porch light"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-6", "double-click-hold").await;

    assert!(
        shield.received_requests().await.unwrap().is_empty(),
        "the classifier must not be asked"
    );
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    let body = &notes[0].body;
    assert!(
        body.contains("home/unlock_door {\"door\":\"back\"}\n  Always asks you (policy ask)"),
        "{body}"
    );
    assert!(body.contains("home/set_light {\"name\":\"porch\",\"on\":true}\n  Not checked: another action here always asks you"), "{body}");
    assert!(!body.contains("delete_everything"), "{body}");
    // The denied tool was not even offered.
    let first = h.claude_seen.lock().unwrap()[0].clone();
    let names: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["home__set_light", "home__unlock_door"]);

    // Approving runs the two allowed calls; the denied one stays blocked.
    let (status, _) = send(
        &h,
        Request::post(button(&notes[0], true))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    tick(&h).await;
    assert_eq!(
        *h.mcp_calls.lock().unwrap(),
        vec![
            "unlock_door back".to_string(),
            "set_light porch true".to_string()
        ]
    );
    assert!(shield.received_requests().await.unwrap().is_empty());
    let seen = h.claude_seen.lock().unwrap().clone();
    let results = &seen[1]["messages"][2]["content"];
    assert_eq!(results[0]["is_error"], true);
    assert!(
        results[0]["content"]
            .as_str()
            .unwrap()
            .contains("Unknown tool home__delete_everything")
    );
}

#[tokio::test]
async fn shadow_mode_records_the_verdict_and_still_asks() {
    let shield = shield_server(|_| shield_reply(0.01)).await;
    let h = harness_config(
        &approval_config(&shield.uri(), "shadow", "confirm", ""),
        vec![
            tool_use_turn("toolu_9", "home__unlock_door", json!({"door": "front"})),
            final_turn("OK, I left it locked."),
        ],
        Some("unlock the front door"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-7", "double-click-hold").await;

    assert_eq!(shield.received_requests().await.unwrap().len(), 1);
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].title, "Ringmaster: approve?");
    assert!(
        notes[0]
            .body
            .contains("(shadow) Classifier would auto-approve this (score 0.01)"),
        "{}",
        notes[0].body
    );
    let c = &audit_of(&h, "classification").await[0];
    assert_eq!(
        (c["verdict"].as_str(), c["outcome"].as_str()),
        (Some("safe"), Some("shadow"))
    );

    // The human denies what the classifier would have run: a disagreement.
    send(
        &h,
        Request::post(button(&notes[0], false))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    tick(&h).await;
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let cmp = &audit_of(&h, "shadow_comparison").await[0];
    assert_eq!(cmp["decision"], "deny");
    assert_eq!(cmp["disagreement"], true);
    assert_eq!(cmp["result"], "classifier_would_have_run_a_denied_action");
    assert_eq!(cmp["judgements"][0]["tool"], "home/unlock_door");
    // Nothing ran on the classifier's say.
    assert_eq!(h.notes.take().pop().unwrap().body, "OK, I left it locked.");
}

#[tokio::test]
async fn one_flagged_call_sends_the_whole_turn_to_the_human() {
    let shield = shield_server(|body| {
        shield_reply(if body.contains("unlock_door") {
            0.9
        } else {
            0.01
        })
    })
    .await;
    let h = harness_config(
        &approval_config(&shield.uri(), "classifier", "confirm", ""),
        vec![multi_tool_turn(&[
            (
                "toolu_l",
                "home__set_light",
                json!({"name": "porch", "on": true}),
            ),
            ("toolu_u", "home__unlock_door", json!({"door": "front"})),
        ])],
        Some("turn on the porch light"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-8", "double-click-hold").await;

    // Each call was classified on its own...
    assert_eq!(shield.received_requests().await.unwrap().len(), 2);
    // ...and the safe one did not run either.
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    let body = &notes[0].body;
    assert!(
        body.contains("Classifier passed this (score 0.01); asking because of the other actions"),
        "{body}"
    );
    assert!(
        body.contains("Classifier flagged this (score 0.90 > 0.50)"),
        "{body}"
    );
    let verdicts: Vec<String> = audit_of(&h, "classification")
        .await
        .iter()
        .map(|c| {
            format!(
                "{} {}",
                c["verdict"].as_str().unwrap(),
                c["outcome"].as_str().unwrap()
            )
        })
        .collect();
    assert_eq!(verdicts, ["safe escalated", "unsafe escalated"]);
}

#[tokio::test]
async fn profile_can_keep_manual_approval_in_classifier_mode() {
    let shield = shield_server(|_| shield_reply(0.01)).await;
    let config = approval_config(&shield.uri(), "classifier", "confirm", "").replace(
        "tools = [\"home/set_light\"]\nsystem_prompt = \"Capture mode.\"",
        "tools = [\"home/set_light\"]\nsystem_prompt = \"Capture mode.\"\napproval_mode = \"manual\"",
    );
    assert!(config.contains("approval_mode = \"manual\""));
    let h = harness_config(
        &config,
        vec![multi_tool_turn(&[(
            "toolu_l",
            "home__set_light",
            json!({"name": "porch", "on": true}),
        )])],
        Some("porch light on"),
        |home| vec![home],
    )
    .await;
    run_command(&h, "rec-shield-9", "single-click-hold").await;
    assert!(shield.received_requests().await.unwrap().is_empty());
    assert!(h.mcp_calls.lock().unwrap().is_empty());
    let notes = h.notes.take();
    assert!(matches!(notes[0].kind, Kind::Confirm { .. }), "{notes:?}");
    assert!(!notes[0].body.contains("Classifier"), "{}", notes[0].body);
}

#[tokio::test]
async fn configs_without_approval_ask_exactly_as_before() {
    let h = harness(
        vec![tool_use_turn(
            "toolu_9",
            "home__unlock_door",
            json!({"door": "front"}),
        )],
        Some("unlock the front door"),
    )
    .await;
    run_command(&h, "rec-shield-10", "double-click-hold").await;
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].title, "Ringmaster: approve?");
    assert_eq!(
        notes[0].body,
        "Request: \"unlock the front door\"\nRun this action:\n\
         home/unlock_door {\"door\":\"front\"}\n(expires in 10 min)"
    );
    assert!(audit_of(&h, "classification").await.is_empty());
    // The parked state has the same shape as before the classifier.
    let state = serde_json::to_value(h.db.get_job(1).await.unwrap().unwrap().agent_state).unwrap();
    assert!(state.get("auto_approved").is_none(), "{state}");
    let pending = &state["pending"];
    assert_eq!(
        pending.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["calls"]
    );
}

// ------------------------------------- argument policies and spoken codes

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Target {
    entity_id: String,
}

/// Home Assistant's `call_action` arguments (helpers in
/// `homeassistant/components/homeassistant/llm.py`).
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ActionArgs {
    /// The action to call, e.g. 'light.turn_on'.
    action: String,
    target: Option<Target>,
}

#[derive(Clone)]
struct Ha {
    calls: Arc<Mutex<Vec<String>>>,
}

#[tool_router]
impl Ha {
    #[tool(description = "Call a Home Assistant action")]
    async fn call_action(
        &self,
        Parameters(a): Parameters<ActionArgs>,
    ) -> Result<CallToolResult, McpError> {
        let entity = a.target.map(|t| t.entity_id).unwrap_or_default();
        self.calls
            .lock()
            .unwrap()
            .push(format!("{} {entity}", a.action));
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "{\"success\": true}",
        )]))
    }
}

#[tool_handler]
impl ServerHandler for Ha {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}

/// A Home Assistant-like MCP server ("homeassistant") with `call_action`.
async fn start_ha() -> (McpServerConfig, Arc<Mutex<Vec<String>>>, CancellationToken) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let ha = Ha {
        calls: calls.clone(),
    };
    let ct = CancellationToken::new();
    let service = StreamableHttpService::new(
        move || Ok(ha.clone()),
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
    let cfg = McpServerConfig {
        name: "homeassistant".into(),
        url: format!("http://{addr}/mcp"),
        ..Default::default()
    };
    (cfg, calls, ct)
}

const HOUSE: &str = "alarm_control_panel.64_104058";
const UNIFI: &str = "alarm_control_panel.loft_dream_machine_pro_alarm_manager";

/// Rob's alarm rules, a garage door on a second code, and the codes.
const ALARM_RULES: &str = r#"
[auth_codes]
lockout_secs = 900

[auth_codes.codes.alarm]
code_env = "RINGMASTER_CODE_ALARM"

[auth_codes.codes.garage]
code_env = "RINGMASTER_CODE_GARAGE"

[[tool_policies]]
match = "homeassistant/*call_action"
when_any = { "/target/entity_id" = "*loft_dream_machine_pro_alarm_manager*", "/data/entity_id" = "*loft_dream_machine_pro_alarm_manager*" }
policy = "deny"

[[tool_policies]]
match = "homeassistant/*call_action"
when = { "/action" = "alarm_control_panel.*" }
when_any = { "/target/area_id" = "*", "/target/device_id" = "*", "/target/floor_id" = "*", "/target/label_id" = "*", "/data/entity_id" = "*", "/data/area_id" = "*", "/data/device_id" = "*", "/data/floor_id" = "*", "/data/label_id" = "*" }
policy = "deny"

[[tool_policies]]
match = "homeassistant/*call_action"
when = { "/action" = "alarm_control_panel.*" }
unless = { "/target/entity_id" = "{alarm_control_panel.64_104058,alarm_control_panel.64_104058_perimeter_only,alarm_control_panel.64_104058_motions_only,alarm_control_panel.64_104058_shed_only}" }
policy = "deny"

[[tool_policies]]
match = "homeassistant/*call_action"
when = { "/action" = "alarm_control_panel.alarm_disarm" }
policy = "code"
code = "alarm"
ON_MISSING

[[tool_policies]]
match = "homeassistant/*call_action"
when = { "/action" = "cover.open_cover" }
policy = "code"
code = "garage"

[[tool_policies]]
match = "homeassistant/*call_action"
when = { "/action" = "alarm_control_panel.*" }
policy = "confirm"

[[tool_policies]]
match = "homeassistant/*call_action"
when = { "/action" = "{script,automation}.*" }
policy = "ask"

[[tool_policies]]
match = "homeassistant/*render_template"
policy = "confirm"

[[tool_policies]]
match = "homeassistant/*"
policy = "allow"
"#;

/// The test config plus `ALARM_RULES` (`on_missing` as given) and `extra`.
fn alarm_config(on_missing: &str, extra: &str) -> String {
    format!(
        "{}\n{}\n{extra}\n",
        config_agent("worker_concurrency = 1"),
        ALARM_RULES.replace("ON_MISSING", on_missing)
    )
}

struct AlarmHarness {
    h: Harness,
    ha_calls: Arc<Mutex<Vec<String>>>,
    _ha_stop: CancellationToken,
}

async fn alarm_harness(config: &str, turns: Vec<Value>, whisper: Option<&str>) -> AlarmHarness {
    let (ha, ha_calls, ha_stop) = start_ha().await;
    let h = harness_config(config, turns, whisper, move |home| vec![home, ha]).await;
    AlarmHarness {
        h,
        ha_calls,
        _ha_stop: ha_stop,
    }
}

fn disarm(id: &str, entity: &str) -> Value {
    tool_use_turn(
        id,
        "homeassistant__call_action",
        json!({"action": "alarm_control_panel.alarm_disarm", "target": {"entity_id": entity}}),
    )
}

async fn send_command(h: &Harness, delivery: &str, phone: Option<&str>) {
    let (ct, body) = multipart(Some(TONE), phone, false);
    let (status, text) = send(
        h,
        signed_request(delivery, "double-click-hold", false, &ct, body, now()),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
}

/// The text of the first user message the model got in request `n`.
fn user_text(h: &Harness, n: usize) -> String {
    let seen = h.claude_seen.lock().unwrap();
    seen[n]["messages"][0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

/// The tool result the model got in request `n`.
fn tool_result(h: &Harness, n: usize) -> Value {
    let seen = h.claude_seen.lock().unwrap();
    seen[n]["messages"][2]["content"][0].clone()
}

/// Everything Ringmaster stored, as one string: every column of every
/// table (transcripts, deliveries, audit rows, saved agent states,
/// confirmations).
fn database_dump(h: &Harness) -> String {
    let conn = rusqlite::Connection::open(&h.db_path).unwrap();
    let mut out = String::new();
    for table in [
        "deliveries",
        "jobs",
        "audit",
        "confirmations",
        "auth_code_failures",
        "auth_code_lockouts",
    ] {
        let mut st = conn.prepare(&format!("SELECT * FROM {table}")).unwrap();
        let n = st.column_count();
        let rows = st
            .query_map([], |r| {
                Ok((0..n)
                    .map(|i| match r.get_ref(i).unwrap() {
                        rusqlite::types::ValueRef::Text(t) => {
                            String::from_utf8_lossy(t).into_owned()
                        }
                        other => format!("{other:?}"),
                    })
                    .collect::<Vec<_>>()
                    .join("|"))
            })
            .unwrap();
        for row in rows {
            out.push_str(&row.unwrap());
            out.push('\n');
        }
    }
    out
}

/// None of the code's words appear in `text`, in any case.
fn assert_no_code(what: &str, text: &str) {
    let lower = text.to_lowercase();
    for word in ["delta", "beta", "better", "echo foxtrot"] {
        assert!(!lower.contains(word), "{what} holds {word:?}: {text}");
    }
}

fn audio_files(h: &Harness) -> usize {
    std::fs::read_dir(h._dir.path().join("audio"))
        .map(|d| d.count())
        .unwrap_or(0)
}

#[tokio::test]
async fn spoken_code_authorises_the_disarm_without_ntfy_or_classifier() {
    for (n, said) in [
        "Disarm the house alarm, authorisation code Delta Beta 5",
        "Disarm the house alarm. Authorisation code delta, beta, five.",
        "Authorization code delta beta five. Disarm the house alarm.",
        "Disarm the house alarm, auth code delta better five",
    ]
    .into_iter()
    .enumerate()
    {
        let shield = shield_server(|_| shield_reply(0.01)).await;
        let config = alarm_config(
            "",
            &format!(
                "[approval]\nmode = \"classifier\"\nbase_url = \"{}\"\nrequest_timeout_secs = 1\nmax_retries = 0\n",
                shield.uri()
            ),
        );
        let a = alarm_harness(
            &config,
            vec![
                disarm("toolu_1", HOUSE),
                final_turn("The house alarm is disarmed."),
            ],
            Some(said),
        )
        .await;
        let h = &a.h;
        // The phone heard the code too.
        send_command(h, &format!("rec-code-{n}"), Some(said)).await;
        tick(h).await;

        assert_eq!(
            *a.ha_calls.lock().unwrap(),
            vec![format!("alarm_control_panel.alarm_disarm {HOUSE}")],
            "{said}"
        );
        let notes = h.notes.take();
        assert_eq!(notes.len(), 1, "no Approve/Deny: {notes:?}");
        assert_eq!(notes[0].kind, Kind::Result);
        assert!(
            notes[0].body.starts_with(
                "The house alarm is disarmed.\n\nAuthorised by code: homeassistant/call_action"
            ),
            "{}",
            notes[0].body
        );
        assert!(
            notes[0].body.contains("(code \"alarm\")"),
            "{}",
            notes[0].body
        );
        assert!(
            shield.received_requests().await.unwrap().is_empty(),
            "classifier asked"
        );
        assert!(
            user_text(h, 0).contains("[authorisation code accepted]"),
            "{said}"
        );
        let call = audit_of(h, "tool_call").await;
        assert_eq!(call[0]["approved_by"], "auth_code");
        assert_eq!(call[0]["auth_code"], "alarm");
        assert_eq!(call[0]["outcome"], "ok");

        // The code is nowhere: database, model requests, notifications.
        let dump = database_dump(h);
        assert!(dump.contains("Disarm the house"), "{dump}");
        assert_no_code("database", &dump);
        for body in h.claude_seen.lock().unwrap().iter() {
            assert_no_code("model request", &body.to_string());
        }
        assert_no_code("notification", &format!("{notes:?}"));
        let job = h.db.get_job(1).await.unwrap().unwrap();
        assert_no_code(
            "agent state",
            &serde_json::to_string(&job.agent_state).unwrap(),
        );
        assert!(
            job.transcript
                .unwrap()
                .contains("[authorisation code accepted]")
        );
        // The recording is gone, whatever the retention.
        assert_eq!(audio_files(h), 0);
        assert!(job.audio_path.is_none());
        let deleted = audit_of(h, "audio_deleted").await;
        assert_eq!(deleted[0]["reason"], "contained an authorisation code");
        let auth = audit_of(h, "auth_code").await;
        assert_eq!(auth[0]["status"], "accepted");
    }
}

#[tokio::test]
async fn wrong_or_missing_code_is_refused_with_a_relayable_error() {
    for (n, said) in [
        "Disarm the house alarm, authorisation code delta beta six",
        "Disarm the house alarm",
    ]
    .into_iter()
    .enumerate()
    {
        let shield = shield_server(|_| shield_reply(0.01)).await;
        let config = alarm_config(
            "",
            &format!(
                "[approval]\nmode = \"classifier\"\nbase_url = \"{}\"\nrequest_timeout_secs = 1\nmax_retries = 0\n",
                shield.uri()
            ),
        );
        let a = alarm_harness(
            &config,
            vec![
                disarm("toolu_1", HOUSE),
                final_turn("Disarming needs your authorisation code."),
            ],
            Some(said),
        )
        .await;
        let h = &a.h;
        send_command(h, &format!("rec-wrong-{n}"), None).await;
        tick(h).await;

        assert!(a.ha_calls.lock().unwrap().is_empty(), "{said}");
        let result = tool_result(h, 1);
        assert_eq!(result["is_error"], true);
        assert!(
            result["content"]
                .as_str()
                .unwrap()
                .contains("needs the user's spoken authorisation code"),
            "{result}"
        );
        let notes = h.notes.take();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].body, "Disarming needs your authorisation code.");
        assert!(shield.received_requests().await.unwrap().is_empty());
        let marker = n == 0;
        assert_eq!(
            user_text(h, 0).contains("[authorisation code rejected]"),
            marker,
            "{said}"
        );
        assert_no_code("database", &database_dump(h));
        assert_no_code("model request", &user_text(h, 0));
        // A recording without a code keeps its audio.
        assert_eq!(audio_files(h), usize::from(!marker));
    }
}

#[tokio::test]
async fn missing_code_can_fall_back_to_asking_but_never_the_classifier() {
    let shield = shield_server(|_| shield_reply(0.01)).await;
    let config = alarm_config(
        "on_missing = \"ask\"",
        &format!(
            "[approval]\nmode = \"classifier\"\nbase_url = \"{}\"\nrequest_timeout_secs = 1\nmax_retries = 0\n",
            shield.uri()
        ),
    );
    let a = alarm_harness(
        &config,
        vec![disarm("toolu_1", HOUSE), final_turn("Disarmed.")],
        Some("Disarm the house alarm, authorisation code delta beta six"),
    )
    .await;
    let h = &a.h;
    send_command(h, "rec-ask", None).await;
    tick(h).await;
    assert!(a.ha_calls.lock().unwrap().is_empty());
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].title, "Ringmaster: approve?");
    assert!(
        notes[0].body.contains("Always asks you (policy ask)"),
        "{}",
        notes[0].body
    );
    assert!(
        notes[0].body.contains("[authorisation code rejected]"),
        "{}",
        notes[0].body
    );
    assert_no_code("confirmation", &notes[0].body);
    assert!(shield.received_requests().await.unwrap().is_empty());

    // Approved by the human, it runs (and is not listed as code-authorised).
    send(
        h,
        Request::post(button(&notes[0], true))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    tick(h).await;
    assert_eq!(a.ha_calls.lock().unwrap().len(), 1);
    let notes = h.notes.take();
    assert_eq!(notes[0].body, "Disarmed.");
    assert_eq!(audit_of(h, "tool_call").await[0]["approved_by"], "human");
}

#[tokio::test]
async fn a_code_authorises_neither_another_job_nor_another_code() {
    // Whisper is down: each job uses its phone transcription.
    let a = alarm_harness(
        &alarm_config("", ""),
        vec![
            tool_use_turn(
                "toolu_g",
                "homeassistant__call_action",
                json!({"action": "cover.open_cover", "target": {"entity_id": "cover.garage"}}),
            ),
            final_turn("The garage needs its own code."),
            disarm("toolu_d", HOUSE),
            final_turn("Disarming needs your authorisation code."),
        ],
        None,
    )
    .await;
    let h = &a.h;
    // The alarm's code, used for the garage.
    send_command(
        h,
        "rec-a",
        Some("open the garage, authorisation code delta beta five"),
    )
    .await;
    tick(h).await;
    // The next request, without a code.
    send_command(h, "rec-b", Some("disarm the house alarm")).await;
    tick(h).await;

    assert!(a.ha_calls.lock().unwrap().is_empty());
    for n in [1, 3] {
        let r = tool_result(h, n);
        assert_eq!(r["is_error"], true, "{r}");
        assert!(
            r["content"]
                .as_str()
                .unwrap()
                .contains("authorisation code"),
            "{r}"
        );
    }
    assert!(user_text(h, 0).contains("[authorisation code accepted]"));
    assert!(!user_text(h, 2).contains("authorisation code"));
    let second = h.db.get_job(2).await.unwrap().unwrap();
    assert!(second.code_auth.unwrap().until.is_empty());
    assert_no_code("database", &database_dump(h));
}

#[tokio::test]
async fn the_unifi_panel_is_refused_even_with_a_valid_code() {
    let a = alarm_harness(
        &alarm_config("", ""),
        vec![
            disarm("toolu_1", UNIFI),
            final_turn("I can't use that panel."),
        ],
        Some("Disarm the loft alarm, authorisation code delta beta five"),
    )
    .await;
    let h = &a.h;
    send_command(h, "rec-unifi", None).await;
    tick(h).await;
    assert!(a.ha_calls.lock().unwrap().is_empty());
    // The tool is offered (the deny depends on the arguments) and the call
    // is refused when made.
    let first = h.claude_seen.lock().unwrap()[0].clone();
    assert!(
        first["tools"]
            .to_string()
            .contains("homeassistant__call_action")
    );
    let r = tool_result(h, 1);
    assert_eq!(r["is_error"], true);
    assert!(
        r["content"]
            .as_str()
            .unwrap()
            .contains("policy blocks this call with these arguments"),
        "{r}"
    );
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1);
    assert!(
        !notes[0].body.contains("Authorised by code"),
        "{}",
        notes[0].body
    );
}

#[tokio::test]
async fn wrong_codes_lock_out_alert_survive_a_restart_and_expire() {
    let a = alarm_harness(
        &alarm_config("", ""),
        vec![
            final_turn("Wrong code."),
            final_turn("Wrong code."),
            final_turn("Wrong code."),
            disarm("toolu_l", HOUSE),
            final_turn("Codes are locked."),
            disarm("toolu_ok", HOUSE),
            final_turn("Disarmed."),
        ],
        None,
    )
    .await;
    let h = &a.h;
    for n in 0..3 {
        send_command(
            h,
            &format!("rec-guess-{n}"),
            Some(&format!("disarm the alarm, code alpha alpha {n}")),
        )
        .await;
        tick(h).await;
    }
    h.notes.take();
    h.watchdog.check(now()).await;
    let sent = alerts(h);
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(
        sent[0].title,
        "Ringmaster alert: authorisation codes locked out"
    );
    assert!(
        sent[0].body.contains("3 wrong authorisation codes"),
        "{}",
        sent[0].body
    );
    assert!(
        !sent[0].body.to_lowercase().contains("alpha"),
        "{}",
        sent[0].body
    );
    assert_eq!(audit_of(h, "auth_code").await[0]["failures"], 1);

    // While locked out the right code is not even checked.
    send_command(
        h,
        "rec-locked",
        Some("disarm the alarm, code delta beta five"),
    )
    .await;
    tick(h).await;
    assert!(a.ha_calls.lock().unwrap().is_empty());
    assert!(
        user_text(h, 3).contains("not checked: locked"),
        "{}",
        user_text(h, 3)
    );
    let r = tool_result(h, 4);
    assert!(r["content"].as_str().unwrap().contains("locked"), "{r}");
    h.notes.take();

    // A restart keeps the lockout and its alert.
    let reopened = Db::open(&h.db_path).unwrap();
    let lockout = reopened.active_lockout(now()).await.unwrap().unwrap();
    let config = Config::from_toml(&alarm_config("", "")).unwrap();
    let recorder = Arc::new(Recorder::default());
    let fresh = Watchdog::new(
        Arc::new(Health::new(&config, now())),
        reopened,
        Alerter::new(recorder.clone(), &config.alerts, AlertBook::load(None)),
    );
    fresh.check(now()).await;
    let again = recorder.take();
    assert!(
        again
            .iter()
            .any(|n| n.title.contains("authorisation codes locked out")),
        "{again:?}"
    );

    // It ends on time: the alert recovers and codes work again.
    h.watchdog.check(lockout.until).await;
    let recovered = alerts(h);
    assert!(
        recovered
            .iter()
            .any(|n| n.kind == Kind::Recovered && n.title.contains("authorisation codes")),
        "{recovered:?}"
    );
    let conn = rusqlite::Connection::open(&h.db_path).unwrap();
    conn.execute("UPDATE auth_code_lockouts SET until = since", [])
        .unwrap();
    send_command(
        h,
        "rec-after",
        Some("disarm the alarm, code delta beta five"),
    )
    .await;
    tick(h).await;
    assert_eq!(a.ha_calls.lock().unwrap().len(), 1);
    assert_no_code("database", &database_dump(h));
}

#[tokio::test]
async fn without_codes_transcripts_and_recordings_are_left_alone() {
    let h = harness(
        vec![final_turn("Card added.")],
        Some("Add a card for the code review, code delta beta five"),
    )
    .await;
    run_command(&h, "rec-plain", "double-click-hold").await;
    assert!(
        user_text(&h, 0).contains("Add a card for the code review, code delta beta five"),
        "{}",
        user_text(&h, 0)
    );
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert!(job.code_auth.is_none() && job.audio_path.is_some());
    assert_eq!(audio_files(&h), 1);
    assert!(audit_of(&h, "audio_deleted").await.is_empty());
}

#[tokio::test]
async fn classifier_requests_never_carry_the_code() {
    let shield = shield_server(|_| shield_reply(0.01)).await;
    let config = alarm_config(
        "",
        &format!(
            "[approval]\nmode = \"classifier\"\nbase_url = \"{}\"\nauto_approve_below = 0.2\nrequest_timeout_secs = 1\nmax_retries = 0\n",
            shield.uri()
        ),
    );
    let a = alarm_harness(
        &config,
        vec![
            multi_tool_turn(&[
                (
                    "toolu_d",
                    "homeassistant__call_action",
                    json!({"action": "alarm_control_panel.alarm_disarm", "target": {"entity_id": HOUSE}}),
                ),
                (
                    "toolu_a",
                    "homeassistant__call_action",
                    json!({"action": "alarm_control_panel.alarm_arm_away",
                           "target": {"entity_id": "alarm_control_panel.64_104058_shed_only"}}),
                ),
            ]),
            final_turn("House disarmed, shed armed."),
        ],
        Some("Disarm the house and arm the shed, authorisation code delta beta five"),
    )
    .await;
    let h = &a.h;
    send_command(
        h,
        "rec-both",
        Some("disarm the house and arm the shed authorisation code delta beta 5"),
    )
    .await;
    tick(h).await;
    assert_eq!(a.ha_calls.lock().unwrap().len(), 2);
    // Only the `confirm` call was classified, and its request has the
    // marker, not the code.
    let requests = shield.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = String::from_utf8_lossy(&requests[0].body).to_string();
    assert!(body.contains("alarm_arm_away"), "{body}");
    assert!(body.contains("[authorisation code accepted]"), "{body}");
    assert_no_code("classifier request", &body);
    let notes = h.notes.take();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(
        notes[0]
            .body
            .contains("Authorised by code: homeassistant/call_action"),
        "{}",
        notes[0].body
    );
    assert!(
        notes[0].body.contains("Auto-approved by the classifier"),
        "{}",
        notes[0].body
    );
    let calls = audit_of(h, "tool_call").await;
    let by: Vec<&str> = calls
        .iter()
        .map(|c| c["approved_by"].as_str().unwrap())
        .collect();
    assert_eq!(by, ["auth_code", "classifier"]);
    assert_no_code("database", &database_dump(h));
}

#[tokio::test]
async fn a_recording_with_a_code_is_deleted_even_if_the_job_is_too_old_to_run() {
    let a = alarm_harness(&alarm_config("", ""), vec![], Some("disarm the alarm")).await;
    let h = &a.h;
    // Recorded two hours ago: past the default max age. (`multipart`
    // stamps the recording time itself, so the body is built here.)
    let old = (jiff::Timestamp::now().as_millisecond() - 7_200_000).to_string();
    let boundary = "b";
    let mut raw = Vec::new();
    raw.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"r.m4a\"\r\nContent-Type: audio/mp4\r\n\r\n").as_bytes(),
    );
    raw.extend_from_slice(TONE);
    raw.extend_from_slice(
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"transcription\"\r\n\r\ndisarm the alarm, code delta beta five\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"recordedAt\"\r\n\r\n{old}\r\n--{boundary}--\r\n"
        )
        .as_bytes(),
    );
    let (status, text) = send(
        h,
        signed_request(
            "rec-old",
            "double-click-hold",
            false,
            &format!("multipart/form-data; boundary={boundary}"),
            raw,
            now(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
    assert_eq!(audio_files(h), 1);
    tick(h).await;
    let job = h.db.get_job(1).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Failed);
    assert!(h.claude_seen.lock().unwrap().is_empty());
    assert_eq!(audio_files(h), 0);
    assert_eq!(audit_of(h, "audio_deleted").await.len(), 1);
    assert_no_code("database", &database_dump(h));
}
