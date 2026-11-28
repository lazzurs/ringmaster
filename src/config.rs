//! Configuration: a TOML file for everything that is not secret, environment
//! variables for everything that is.
//!
//! Every secret can also be given as `<NAME>_FILE` pointing at a file holding
//! the value (Docker / Compose secrets).

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::llm::Provider;
use crate::policy::Policy;

pub const ENV_ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";
pub const ENV_WEBHOOK_SECRET: &str = "RINGMASTER_WEBHOOK_SECRET";
pub const ENV_NTFY_TOKEN: &str = "NTFY_TOKEN";

/// Frozen preamble that every profile's system prompt is appended to. It is part
/// of the cached prompt prefix, so it must never contain per-request data.
pub const DEFAULT_BASE_PROMPT: &str = "\
You are Ringmaster, a voice assistant that carries out short spoken requests captured by a smart ring. \
The request arrives as an automatic speech-to-text transcript and may contain recognition errors: \
interpret it charitably, using the names of known devices, rooms, boards and lists. \
Use the provided tools to carry out the request. Prefer acting over asking; if the request is \
ambiguous, unsafe, or no tool fits, do nothing and say so briefly. \
If the user declines a tool call, do not try to achieve the same effect another way. \
Your final reply is shown as a phone notification: one or two short plain-text sentences saying \
what you did or what went wrong. No markdown, no lists, no preamble.";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub webhook: WebhookConfig,
    #[serde(default)]
    pub stt: SttConfig,
    /// Provider-neutral model settings. See [`Config::provider_settings`]
    /// and [`Config::profile_llm`] for how it combines with `[claude]`.
    #[serde(default)]
    pub llm: LlmConfig,
    /// Legacy Claude settings, still honoured: they are the Anthropic
    /// defaults under `[llm]` / `[llm.anthropic]`.
    #[serde(default)]
    pub claude: ClaudeConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub retention: RetentionConfig,
    #[serde(default)]
    pub notify: NotifyConfig,
    /// Operational alerts (database, stalled queue) and the health checks
    /// behind them.
    #[serde(default)]
    pub alerts: AlertsConfig,
    /// How `confirm` tool calls get approved: by you (default), or by a
    /// local Shieldstral classifier with you as the fallback.
    #[serde(default)]
    pub approval: ApprovalConfig,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    #[serde(default)]
    pub tool_policies: Vec<ToolPolicyRule>,
    /// Spoken authorisation codes for `code` tool policies.
    #[serde(default)]
    pub auth_codes: AuthCodesConfig,
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileConfig>,
    /// Problems with individual `[[mcp_servers]]` entries. Those entries are
    /// dropped instead of stopping the service; `check-config` reports them.
    #[serde(skip)]
    pub mcp_warnings: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    /// Address the HTTP server listens on.
    pub listen: SocketAddr,
    /// Externally reachable base URL (e.g. the Cloudflare Tunnel hostname).
    /// Used to build the Approve/Deny links in confirmation notifications.
    pub public_base_url: Option<String>,
    /// Directory holding the SQLite database and stored audio (the only
    /// writable path in the container).
    pub data_dir: PathBuf,
    /// Maximum accepted webhook body size in bytes.
    pub max_body_bytes: usize,
    /// On SIGTERM: how long in-flight requests and jobs get to finish before
    /// the process exits anyway (unfinished jobs are requeued at the next
    /// start). Keep it below the container's stop timeout (Docker: 10 s
    /// unless `stop_grace_period` says otherwise).
    pub shutdown_grace_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".parse().expect("static address"),
            public_base_url: None,
            data_dir: PathBuf::from("/data"),
            max_body_bytes: 25 * 1024 * 1024,
            shutdown_grace_secs: 8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WebhookConfig {
    /// Path the ring app posts to.
    pub path: String,
    /// Reject requests without a valid `X-Index-Signature`. Needs
    /// `RINGMASTER_WEBHOOK_SECRET`.
    pub require_signature: bool,
    /// Allowed clock skew for `X-Index-Timestamp`, in seconds.
    pub timestamp_tolerance_secs: i64,
    /// Send a notification when a test event arrives (handy for checking the
    /// whole path from the phone to ntfy).
    pub notify_test_events: bool,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            path: "/webhook".into(),
            require_signature: true,
            timestamp_tolerance_secs: 300,
            notify_test_events: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SttConfig {
    /// whisper.cpp `whisper-server` base URL, e.g. `http://mac-mini.lan:8178`.
    pub primary_url: Option<String>,
    /// Optional second `whisper-server`, tried when the primary fails.
    pub fallback_url: Option<String>,
    /// Path of the inference endpoint on the servers.
    pub inference_path: String,
    /// Spoken language passed to whisper (`auto` to detect).
    pub language: String,
    /// Words whisper should be biased towards: device, room, board names...
    pub vocabulary: Vec<String>,
    /// Explicit initial prompt. When unset one is built from `vocabulary`.
    pub prompt: Option<String>,
    /// Use the phone's own transcription when no local STT server answers.
    pub use_phone_transcription: bool,
    pub connect_timeout_secs: u64,
    pub timeout_secs: u64,
    /// Recordings larger than this are not decoded (the phone's
    /// transcription is used instead).
    pub max_audio_bytes: u64,
    /// Decoding stops (and the phone's transcription is used) once a
    /// recording turns out to be longer than this.
    pub max_audio_secs: u64,
    /// Wall-clock limit for decoding one recording.
    pub decode_timeout_secs: u64,
    /// Recordings decoded at once. Decoding runs on a blocking thread that
    /// cannot be cancelled, so this also caps how many threads a decoder
    /// that never returns can tie up.
    pub decode_concurrency: usize,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            primary_url: None,
            fallback_url: None,
            inference_path: "/inference".into(),
            language: "en".into(),
            vocabulary: Vec::new(),
            prompt: None,
            use_phone_transcription: true,
            connect_timeout_secs: 5,
            timeout_secs: 120,
            max_audio_bytes: 25 * 1024 * 1024,
            max_audio_secs: 600,
            decode_timeout_secs: 60,
            decode_concurrency: 1,
        }
    }
}

impl SttConfig {
    /// The initial prompt sent to whisper to bias recognition.
    pub fn initial_prompt(&self) -> Option<String> {
        if let Some(p) = &self.prompt {
            return (!p.trim().is_empty()).then(|| p.clone());
        }
        if self.vocabulary.is_empty() {
            return None;
        }
        Some(format!(
            "A short voice command to a smart home and task assistant. Names that may come up: {}.",
            self.vocabulary.join(", ")
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FallbackMode {
    /// On for models that support server-side refusal fallbacks, off for
    /// Haiku (which has none and rejects a fallback list).
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ClaudeConfig {
    pub api_base: String,
    /// `claude-opus-5-5` (default), `claude-sonnet-5-5` or `claude-haiku-5-5`.
    pub model: String,
    /// `output_config.effort`: low | medium | high | xhigh | max.
    pub effort: String,
    /// Covers thinking as well as the reply.
    pub max_tokens: u32,
    /// Maximum Messages API calls per job.
    pub max_iterations: u32,
    /// Server-side refusal fallbacks (`fallbacks: "default"`).
    pub refusal_fallbacks: FallbackMode,
    pub request_timeout_secs: u64,
    /// Retries per API call for 429 / 5xx / 529 / connection errors.
    pub max_retries: u32,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            api_base: "https://api.anthropic.com".into(),
            model: "claude-opus-5-5".into(),
            effort: "low".into(),
            max_tokens: 16000,
            max_iterations: 8,
            refusal_fallbacks: FallbackMode::Auto,
            request_timeout_secs: 120,
            max_retries: 4,
        }
    }
}

impl FallbackMode {
    pub fn enabled_for(self, model: &str) -> bool {
        match self {
            FallbackMode::On => true,
            FallbackMode::Off => false,
            FallbackMode::Auto => !model.starts_with("claude-haiku"),
        }
    }
}

impl ClaudeConfig {
    pub fn fallbacks_enabled(&self, model: &str) -> bool {
        self.refusal_fallbacks.enabled_for(model)
    }
}

/// `[llm]`: the default provider and model, plus a table per provider.
///
/// Keys directly under `[llm]` describe the default provider; a
/// `[llm.<provider>]` table configures that provider whether it is the
/// default or only used by some profiles, and wins over `[llm]`. For
/// Anthropic, unset keys fall back to the legacy `[claude]` section.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    /// Default provider (`anthropic` when unset).
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub request_timeout_secs: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_tokens: Option<u32>,
    /// Model calls per job, for every provider.
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub anthropic: ProviderConfig,
    #[serde(default)]
    pub openai: ProviderConfig,
    #[serde(default)]
    pub mistral: ProviderConfig,
    #[serde(default)]
    pub openrouter: ProviderConfig,
    #[serde(default)]
    pub ollama: ProviderConfig,
}

impl LlmConfig {
    pub fn table(&self, p: Provider) -> &ProviderConfig {
        match p {
            Provider::Anthropic => &self.anthropic,
            Provider::OpenAi => &self.openai,
            Provider::Mistral => &self.mistral,
            Provider::OpenRouter => &self.openrouter,
            Provider::Ollama => &self.ollama,
        }
    }
}

/// `[llm.<provider>]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// Model used when a profile picks this provider without naming one.
    pub model: Option<String>,
    pub effort: Option<String>,
    /// Without the `/v1` path, e.g. `http://m1.lan:11434`.
    pub base_url: Option<String>,
    /// Environment variable holding the API key.
    pub api_key_env: Option<String>,
    pub request_timeout_secs: Option<u64>,
    pub max_retries: Option<u32>,
    /// Output limit; unset leaves it to the provider (Anthropic: 16000).
    pub max_tokens: Option<u32>,
    /// Anthropic only.
    pub refusal_fallbacks: Option<FallbackMode>,
    /// OpenRouter only: `HTTP-Referer` attribution header.
    pub app_url: Option<String>,
    /// OpenRouter only: `X-OpenRouter-Title` attribution header.
    pub app_title: Option<String>,
    /// Ollama only: context window in tokens (`options.num_ctx`); 0 leaves
    /// it to the server.
    pub num_ctx: Option<u32>,
}

/// Default context window asked of Ollama. Its own default on a 16 GB Mac is
/// 4096 tokens, too small for a system prompt plus tool definitions.
pub const OLLAMA_DEFAULT_NUM_CTX: u32 = 8192;

/// Everything needed to talk to one provider, resolved from the config.
#[derive(Debug, Clone)]
pub struct ProviderSettings {
    pub provider: Provider,
    pub base_url: String,
    /// Where the key comes from (`None`: no key, Ollama's default).
    pub api_key_env: Option<String>,
    pub request_timeout: std::time::Duration,
    pub max_retries: u32,
    pub max_tokens: Option<u32>,
    pub refusal_fallbacks: FallbackMode,
    pub app_url: Option<String>,
    pub app_title: Option<String>,
    pub num_ctx: Option<u32>,
}

/// Provider, model and effort a job starts with.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmChoice {
    pub provider: Provider,
    pub model: String,
    /// Empty: leave it to the provider.
    pub effort: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AgentConfig {
    /// IANA time zone used for the "current time" given to Claude.
    pub timezone: String,
    /// Replaces [`DEFAULT_BASE_PROMPT`].
    pub base_system_prompt: Option<String>,
    /// How long an Approve/Deny link stays valid.
    pub confirmation_ttl_secs: u64,
    /// Give up on a job (and say so) once it is this old. Profiles may
    /// override it: a stale "unlock the door" should not run hours later.
    pub job_max_age_secs: u64,
    pub retry_base_secs: u64,
    pub retry_max_secs: u64,
    /// Tell the user when a job is first delayed (Mac asleep, API down...).
    pub notify_on_retry: bool,
    pub tool_timeout_secs: u64,
    /// How long to wait for an MCP server's tool list before running the job
    /// without that server.
    pub tool_discovery_timeout_secs: u64,
    /// Tool output longer than this is truncated before it goes to Claude.
    pub max_tool_result_chars: usize,
    /// Wall-clock limit for one processing attempt of a job (transcription,
    /// model calls and tool calls together). An attempt that runs over is
    /// treated like an outage: the job backs off and is retried until its
    /// max age.
    pub attempt_timeout_secs: u64,
    /// Jobs processed at the same time, so one slow or hung job does not
    /// hold up the rest of the queue.
    pub worker_concurrency: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            timezone: "UTC".into(),
            base_system_prompt: None,
            confirmation_ttl_secs: 600,
            job_max_age_secs: 3600,
            retry_base_secs: 15,
            retry_max_secs: 600,
            notify_on_retry: true,
            tool_timeout_secs: 60,
            tool_discovery_timeout_secs: 10,
            max_tool_result_chars: 20_000,
            attempt_timeout_secs: 900,
            worker_concurrency: 2,
        }
    }
}

/// Upper bound for `agent.worker_concurrency`: one SQLite connection and a
/// single host do not gain from more.
pub const MAX_WORKER_CONCURRENCY: usize = 16;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RetentionConfig {
    /// Stored recordings older than this are deleted (0 = keep for ever).
    pub audio_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self { audio_days: 30 }
    }
}

/// `[alerts]`: operational alerts sent to the operator through the notifier.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AlertsConfig {
    /// Send alerts (they are always logged and shown on `/healthz` and
    /// `/metrics`).
    pub enabled: bool,
    /// ntfy topic for alerts; the normal `notify.ntfy.topic` when unset.
    pub ntfy_topic: Option<String>,
    /// How often the watchdog looks at the queue, the worker and the
    /// database.
    pub check_interval_secs: u64,
    /// While a problem lasts, repeat its alert this often (0 = never).
    pub reminder_secs: u64,
    /// How often to run `PRAGMA quick_check` and a write probe on the
    /// database (also run at once after a corruption or storage error).
    pub db_check_interval_secs: u64,
    /// Database statements failing in a row before an alert. Corruption,
    /// a full disk, a read-only database and I/O errors alert at once.
    pub db_error_threshold: u32,
    /// Alert when the worker loop has not run for this long (it runs at
    /// least every 30 s).
    pub worker_stall_secs: u64,
    /// Alert when a due job has waited this long without being started.
    pub queue_stall_secs: u64,
    /// Alert when a job runs this long past `agent.attempt_timeout_secs`
    /// (its deadline did not fire).
    pub overrun_margin_secs: u64,
    /// Alert when this many jobs fail (failed, expired, quarantined or
    /// panicked) within `failure_window_secs`.
    pub failure_threshold: usize,
    pub failure_window_secs: u64,
}

impl Default for AlertsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ntfy_topic: None,
            check_interval_secs: 30,
            reminder_secs: 3600,
            db_check_interval_secs: 300,
            db_error_threshold: 3,
            worker_stall_secs: 120,
            queue_stall_secs: 600,
            overrun_margin_secs: 60,
            failure_threshold: 3,
            failure_window_secs: 900,
        }
    }
}

/// The worker loop runs at least this often; `alerts.worker_stall_secs`
/// must leave room for it.
pub const WORKER_LOOP_MAX_WAIT_SECS: u64 = 30;

/// Upper bound for `stt.decode_concurrency`.
pub const MAX_DECODE_CONCURRENCY: usize = 8;

/// `approval.mode` (and a profile's `approval_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    /// Every `confirm` call waits for your Approve/Deny (the behaviour
    /// before the classifier existed).
    #[default]
    Manual,
    /// As `manual`, and the classifier's verdict on each `confirm` call is
    /// recorded next to your decision, to compare before trusting it.
    Shadow,
    /// `confirm` calls the classifier judges safe run without asking;
    /// everything else (flagged, uncertain, classifier down) asks you.
    Classifier,
}

impl ApprovalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ApprovalMode::Manual => "manual",
            ApprovalMode::Shadow => "shadow",
            ApprovalMode::Classifier => "classifier",
        }
    }
}

/// Default `<Instruct>` for the approval classifier: the context and the
/// strictness. Constant, so it never carries per-request data.
pub const DEFAULT_APPROVAL_INSTRUCT: &str = "\
You are a strict safety reviewer for a voice assistant that controls a smart home and personal \
task boards. The Document holds a spoken request (an automatic transcript that may contain \
recognition errors) and one tool call the assistant proposes to carry it out. Judge the proposed \
action against the request. Watch for: doing more than was asked, acting on a different device, \
room, board, card or person than the one named, harm to people or property, deleting or \
overwriting data, security (locks, alarms, doors, access), privacy, and anything the user would \
not expect. Apply a low tolerance threshold: if the action does not clearly match the request, \
answer yes.";

/// Default `<Query>`: "yes" means unsafe (escalate to a human).
pub const DEFAULT_APPROVAL_QUERY: &str = "\
Does this proposed action go beyond what the user asked for, or could it cause harm, data loss, \
a security or privacy risk, or an action the user would not expect?";

/// `[approval]`: automatic approval of `confirm` tool calls by a local
/// Shieldstral classifier (an OpenAI-compatible `/v1/chat/completions`
/// server: llama.cpp `llama-server`, vLLM, SGLang). The classifier can
/// approve but never deny: anything it does not clearly pass, and any
/// failure to get a verdict, goes to the normal Approve/Deny notification.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ApprovalConfig {
    /// `manual` (default), `shadow` or `classifier`; profiles may override
    /// it with `approval_mode`.
    pub mode: ApprovalMode,
    /// Classifier server, without the `/v1` path, e.g.
    /// `http://m1-mac.lan:8090`. Required unless every mode is `manual`.
    pub base_url: Option<String>,
    /// Model name sent in the request (vLLM checks it; llama-server
    /// ignores it).
    pub model: String,
    /// Environment variable holding a bearer token for the server, if it
    /// needs one.
    pub api_key_env: Option<String>,
    /// How long to wait for a verdict, retries included, before asking a
    /// human instead.
    pub request_timeout_secs: u64,
    /// Retries for connection errors, 429 and 5xx (within the timeout).
    pub max_retries: u32,
    /// A score above this is "flagged" (the model card's default is 0.5).
    pub threshold: f64,
    /// Only scores below this run without asking; scores from here up to
    /// `threshold` are a grey band that asks a human. Unset: `threshold`.
    pub auto_approve_below: Option<f64>,
    /// Replaces [`DEFAULT_APPROVAL_INSTRUCT`].
    pub instruct: Option<String>,
    /// Replaces [`DEFAULT_APPROVAL_QUERY`]. Phrase it so "yes" means unsafe.
    pub query: Option<String>,
    /// Alert after this many classifier failures in a row.
    pub alert_after_failures: u32,
}

impl Default for ApprovalConfig {
    fn default() -> Self {
        Self {
            mode: ApprovalMode::Manual,
            base_url: None,
            model: "mistralai/Shieldstral-1.0-3B".into(),
            api_key_env: None,
            request_timeout_secs: 15,
            max_retries: 1,
            threshold: 0.5,
            auto_approve_below: None,
            instruct: None,
            query: None,
            alert_after_failures: 3,
        }
    }
}

impl ApprovalConfig {
    /// Scores below this are approved automatically.
    pub fn auto_approve_cutoff(&self) -> f64 {
        self.auto_approve_below.unwrap_or(self.threshold)
    }

    pub fn instruct(&self) -> &str {
        self.instruct
            .as_deref()
            .unwrap_or(DEFAULT_APPROVAL_INSTRUCT)
    }

    pub fn query(&self) -> &str {
        self.query.as_deref().unwrap_or(DEFAULT_APPROVAL_QUERY)
    }
}

/// Upper bound for `approval.request_timeout_secs`: the verdict is awaited
/// inside a job attempt.
pub const MAX_APPROVAL_TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotifyConfig {
    pub ntfy: Option<NtfyConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NtfyConfig {
    /// Server base URL, e.g. `https://ntfy.sh` or `https://ntfy.example.com`.
    #[serde(default = "default_ntfy_url")]
    pub url: String,
    pub topic: String,
    /// Extra headers the phone sends with the Approve/Deny `http` actions.
    /// They are stored by the ntfy server as part of the message, so do not
    /// put long-lived secrets here when using a shared server.
    #[serde(default)]
    pub action_headers: BTreeMap<String, String>,
}

fn default_ntfy_url() -> String {
    "https://ntfy.sh".into()
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerConfig {
    /// Short name used in tool names and policies (`<name>/<tool>`):
    /// lowercase letters, digits and `-`.
    pub name: String,
    /// Streamable HTTP endpoint, e.g. `http://nextcloud-mcp:8000/mcp`.
    pub url: String,
    /// Environment variable holding a bearer token for this server.
    #[serde(default)]
    pub token_env: Option<String>,
    /// HTTP Basic credentials, e.g. a Nextcloud user and app password.
    #[serde(default)]
    pub basic_auth: Option<McpBasicAuth>,
    /// Extra request headers: header name -> environment variable holding
    /// its value (e.g. `{ "X-Api-Key" = "SOME_SERVICE_KEY" }`).
    #[serde(default)]
    pub headers_env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpBasicAuth {
    pub username: String,
    /// Environment variable holding the password.
    pub password_env: String,
}

/// Headers the MCP transport manages itself; they cannot be overridden.
const RESERVED_MCP_HEADERS: &[&str] = &[
    "accept",
    "content-type",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
];

/// MCP server name -> request headers (name, value), credentials included.
pub type McpHeaders = HashMap<String, Vec<(String, String)>>;

/// The request headers for each MCP server, built from its config and the
/// environment. Problems become warnings: the server is then tried without
/// that header, never left out and never fatal.
pub fn mcp_headers(
    servers: &[McpServerConfig],
    lookup: impl Fn(&str) -> Result<Option<String>>,
) -> (McpHeaders, Vec<String>) {
    use base64::Engine;
    let mut out = HashMap::new();
    let mut warnings = Vec::new();
    for s in servers {
        let mut headers: Vec<(String, String)> = Vec::new();
        let mut warn = |msg: String| warnings.push(format!("MCP server {:?}: {msg}", s.name));
        let read = |var: &str, what: &str, warn: &mut dyn FnMut(String)| match lookup(var) {
            Ok(Some(v)) => Some(v),
            Ok(None) => {
                warn(format!("{var} is not set; connecting without {what}"));
                None
            }
            Err(e) => {
                warn(format!("{e:#}; connecting without {what}"));
                None
            }
        };
        if let Some(b) = &s.basic_auth {
            if let Some(pw) = read(&b.password_env, "Basic credentials", &mut warn) {
                let enc = base64::engine::general_purpose::STANDARD
                    .encode(format!("{}:{pw}", b.username));
                headers.push(("Authorization".into(), format!("Basic {enc}")));
            }
            if s.token_env.is_some() {
                warn("token_env is ignored because basic_auth is set".into());
            }
        } else if let Some(var) = &s.token_env
            && let Some(tok) = read(var, "a token", &mut warn)
        {
            headers.push(("Authorization".into(), format!("Bearer {tok}")));
        }
        for (name, var) in &s.headers_env {
            let lower = name.to_ascii_lowercase();
            if RESERVED_MCP_HEADERS.contains(&lower.as_str()) {
                warn(format!(
                    "header {name:?} is managed by the MCP transport; ignored"
                ));
                continue;
            }
            if lower == "authorization" && !headers.is_empty() {
                warn(
                    "headers_env Authorization ignored: basic_auth or token_env already sets it"
                        .into(),
                );
                continue;
            }
            if reqwest::header::HeaderName::from_bytes(name.as_bytes()).is_err() {
                warn(format!("{name:?} is not a valid header name; ignored"));
                continue;
            }
            let Some(value) = read(var, &format!("header {name}"), &mut warn) else {
                continue;
            };
            if reqwest::header::HeaderValue::from_str(&value).is_err() {
                warn(format!(
                    "the value of {var} is not a valid header value; header {name} ignored"
                ));
                continue;
            }
            headers.push((name.clone(), value));
        }
        if !headers.is_empty() {
            out.insert(s.name.clone(), headers);
        }
    }
    (out, warnings)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPolicyRule {
    /// Glob over `<server>/<tool>`, e.g. `homey/*lock*`.
    #[serde(rename = "match")]
    pub pattern: String,
    pub policy: Policy,
    /// JSON pointer into the call's arguments -> glob over the value. The
    /// rule matches only when every one of them matches.
    #[serde(default)]
    pub when: BTreeMap<String, String>,
    /// As `when`, but one of them matching is enough.
    #[serde(default)]
    pub when_any: BTreeMap<String, String>,
    /// The rule does not match when every one of these holds (for a list,
    /// every element must match). A missing pointer does not hold.
    #[serde(default)]
    pub unless: BTreeMap<String, String>,
    /// `policy = "code"`: the `[auth_codes.codes]` entry the call needs
    /// (may be left out when exactly one code is configured).
    #[serde(default)]
    pub code: Option<String>,
    /// `policy = "code"`: what happens without a valid code.
    #[serde(default)]
    pub on_missing: Option<OnMissing>,
}

impl ToolPolicyRule {
    /// A rule on the tool name alone.
    pub fn new(pattern: &str, policy: Policy) -> Self {
        Self {
            pattern: pattern.into(),
            policy,
            when: BTreeMap::new(),
            when_any: BTreeMap::new(),
            unless: BTreeMap::new(),
            code: None,
            on_missing: None,
        }
    }

    /// Whether the rule looks at the arguments at all.
    pub fn is_conditional(&self) -> bool {
        !(self.when.is_empty() && self.when_any.is_empty() && self.unless.is_empty())
    }
}

/// What a `code` rule does when the job has no valid code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OnMissing {
    /// Refuse, with an error result the model can relay.
    #[default]
    Deny,
    /// Ask on the phone (Approve/Deny), as policy `ask`.
    Ask,
}

/// `[auth_codes]`: spoken authorisation codes. Code values come only from
/// the environment (`code_env`, or `<code_env>_FILE`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuthCodesConfig {
    /// Phrases that introduce a code; the code words follow them.
    pub triggers: Vec<String>,
    /// How long after the recording an accepted code stays valid for its
    /// job (a job resumed or retried later needs a new code).
    pub valid_for_secs: u64,
    /// Wrong codes within `failure_window_secs` that lock codes out.
    pub max_failures: u32,
    pub failure_window_secs: u64,
    /// How long codes stay locked out.
    pub lockout_secs: u64,
    /// Named codes.
    pub codes: BTreeMap<String, AuthCodeConfig>,
}

impl Default for AuthCodesConfig {
    fn default() -> Self {
        Self {
            triggers: DEFAULT_CODE_TRIGGERS
                .iter()
                .map(|t| t.to_string())
                .collect(),
            valid_for_secs: 300,
            max_failures: 3,
            failure_window_secs: 600,
            lockout_secs: 900,
            codes: BTreeMap::new(),
        }
    }
}

/// Default `auth_codes.triggers`.
pub const DEFAULT_CODE_TRIGGERS: [&str; 4] = [
    "authorisation code",
    "authorization code",
    "auth code",
    "code",
];

/// One named code.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthCodeConfig {
    /// Environment variable holding the spoken code. Several spoken forms
    /// (what whisper may hear instead) are separated by `|`.
    pub code_env: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfig {
    /// `X-Index-Trigger` values that select this profile.
    pub triggers: Vec<String>,
    /// Appended to the base system prompt.
    pub system_prompt: String,
    /// Globs over `<server>/<tool>` the profile may use. Empty = no tools.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Provider for this profile (default: `llm.provider`).
    #[serde(default)]
    pub provider: Option<Provider>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub job_max_age_secs: Option<u64>,
    /// Overrides `approval.mode` for this profile.
    #[serde(default)]
    pub approval_mode: Option<ApprovalMode>,
}

/// Secrets read from the environment.
#[derive(Clone, Default)]
pub struct Secrets {
    /// API keys of the providers that have one set.
    pub llm_keys: BTreeMap<Provider, String>,
    pub webhook_secret: Option<String>,
    pub ntfy_token: Option<String>,
    /// Bearer token for the approval classifier (`approval.api_key_env`).
    pub classifier_key: Option<String>,
    /// MCP server name -> request headers (credentials included).
    pub mcp_headers: McpHeaders,
    /// MCP token problems; the server is still tried without a token.
    pub warnings: Vec<String>,
}

impl std::fmt::Debug for Secrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Secrets")
            .field("llm_keys", &self.llm_keys.keys().collect::<Vec<_>>())
            .field("webhook_secret", &self.webhook_secret.is_some())
            .field("ntfy_token", &self.ntfy_token.is_some())
            .field("classifier_key", &self.classifier_key.is_some())
            .field("mcp_auth", &self.mcp_headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Read `name`, or the file named by `name_FILE`. Empty values count as unset.
pub fn secret_from_env(name: &str) -> Result<Option<String>> {
    if let Ok(v) = std::env::var(name) {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Ok(Some(v));
        }
    }
    if let Ok(path) = std::env::var(format!("{name}_FILE")) {
        let v = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {name}_FILE ({path})"))?;
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Ok(Some(v));
        }
    }
    Ok(None)
}

/// `value` as a request header carrying a credential: marked sensitive, so
/// it prints as `Sensitive` in `Debug` output (requests, header maps, the
/// clients holding it) and HTTP/2 never adds it to the HPACK table. `None`
/// if it is not a valid header value; the value itself is never reported.
pub fn credential_header(value: &str) -> Option<reqwest::header::HeaderValue> {
    let mut v = reqwest::header::HeaderValue::from_str(value).ok()?;
    v.set_sensitive(true);
    Some(v)
}

/// Secret values every [`redact_credentials`] call removes as well, whatever
/// credentials it is given: the spoken authorisation codes.
static EXTRA_SECRETS: std::sync::RwLock<Vec<String>> = std::sync::RwLock::new(Vec::new());

/// Have every [`redact_credentials`] call remove `value` too (values
/// shorter than 4 characters are ignored, as there).
pub fn register_secret(value: &str) {
    let value = value.trim();
    if value.len() < 4 {
        return;
    }
    let mut extra = EXTRA_SECRETS.write().unwrap_or_else(|p| p.into_inner());
    if !extra.iter().any(|v| v == value) {
        extra.push(value.to_string());
    }
}

/// `text` with every credential replaced by `[redacted]`, for error strings
/// built from what a server sent back (a misbehaving proxy can echo request
/// headers). A value with an auth scheme (`Bearer <token>`) also has the bare
/// token removed. Values shorter than 4 characters are left alone. Secrets
/// registered with [`register_secret`] are always removed.
pub fn redact_credentials<'a>(
    text: &str,
    credentials: impl IntoIterator<Item = &'a str>,
) -> String {
    let extra = EXTRA_SECRETS
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let mut out = text.to_string();
    let mut redact = |needle: &str| {
        if needle.len() >= 4 && out.contains(needle) {
            out = out.replace(needle, "[redacted]");
        }
    };
    for c in credentials {
        let c = c.trim();
        redact(c);
        redact(c.split_once(' ').map_or(c, |(_, t)| t.trim()));
    }
    // Registered secrets (spoken codes) are phrases: only the whole value.
    for c in &extra {
        redact(c);
    }
    out
}

/// A URL for display (`check-config`, error messages) with any password in
/// it replaced by `***`. Text that does not parse as a URL is returned as is.
pub fn display_url(raw: &str) -> String {
    match url::Url::parse(raw) {
        Ok(mut u) if u.password().is_some() => {
            let _ = u.set_password(Some("***"));
            u.to_string()
        }
        _ => raw.to_string(),
    }
}

impl Secrets {
    pub fn from_env(config: &Config) -> Result<Self> {
        let (mcp_headers, mut warnings) = mcp_headers(&config.mcp_servers, secret_from_env);
        let used = config.used_providers();
        let mut llm_keys = BTreeMap::new();
        for p in Provider::ALL {
            let Some(var) = config.provider_settings(p).api_key_env else {
                continue;
            };
            match secret_from_env(&var) {
                Ok(Some(k)) => {
                    llm_keys.insert(p, k);
                }
                Ok(None) => {}
                // A broken key file only matters for a provider in use.
                Err(e) if used.contains(&p) => return Err(e),
                Err(e) => warnings.push(format!("{p}: {e:#}")),
            }
        }
        // A missing or unreadable classifier key only costs the classifier:
        // its requests fail and every call goes to a human, as when it is
        // down.
        let mut classifier_key = None;
        if let Some(var) = &config.approval.api_key_env
            && config.uses_classifier()
        {
            match secret_from_env(var) {
                Ok(Some(k)) => classifier_key = Some(k),
                Ok(None) => warnings.push(format!(
                    "approval: {var} is not set; calling the classifier without a key"
                )),
                Err(e) => warnings.push(format!(
                    "approval: {e:#}; calling the classifier without a key"
                )),
            }
        }
        Ok(Self {
            llm_keys,
            webhook_secret: secret_from_env(ENV_WEBHOOK_SECRET)?,
            ntfy_token: secret_from_env(ENV_NTFY_TOKEN)?,
            classifier_key,
            mcp_headers,
            warnings,
        })
    }
}

impl Config {
    pub fn from_toml(text: &str) -> Result<Self> {
        let mut cfg: Config = toml::from_str(text).context("parsing config")?;
        cfg.prune_mcp_servers();
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        Self::from_toml(&text)
    }

    pub fn base_prompt(&self) -> &str {
        self.agent
            .base_system_prompt
            .as_deref()
            .unwrap_or(DEFAULT_BASE_PROMPT)
    }

    /// The profile name selected by an `X-Index-Trigger` value.
    pub fn profile_for_trigger(&self, trigger: &str) -> Option<&str> {
        self.profiles
            .iter()
            .find(|(_, p)| p.triggers.iter().any(|t| t == trigger))
            .map(|(name, _)| name.as_str())
    }

    /// The default provider (`llm.provider`, else `anthropic`).
    pub fn default_provider(&self) -> Provider {
        self.llm.provider.unwrap_or_default()
    }

    /// Connection settings for a provider. Precedence: `[llm.<provider>]`,
    /// then `[llm]` if it is the default provider, then (Anthropic only)
    /// `[claude]`, then built-in defaults.
    pub fn provider_settings(&self, p: Provider) -> ProviderSettings {
        let t = self.llm.table(p);
        let is_default = p == self.default_provider();
        let anthropic = p == Provider::Anthropic;
        let base_url = t
            .base_url
            .clone()
            .or_else(|| self.llm.base_url.clone().filter(|_| is_default))
            .unwrap_or_else(|| {
                if anthropic {
                    self.claude.api_base.clone()
                } else {
                    p.default_base_url().to_string()
                }
            });
        let api_key_env = t
            .api_key_env
            .clone()
            .or_else(|| self.llm.api_key_env.clone().filter(|_| is_default))
            .or_else(|| p.default_key_env().map(str::to_string));
        // A slow local model needs far longer than a hosted API.
        let (def_timeout, def_retries) = match p {
            Provider::Anthropic => (self.claude.request_timeout_secs, self.claude.max_retries),
            Provider::Ollama => (600, 1),
            _ => (120, 4),
        };
        let request_timeout = std::time::Duration::from_secs(
            t.request_timeout_secs
                .or_else(|| self.llm.request_timeout_secs.filter(|_| is_default))
                .unwrap_or(def_timeout),
        );
        let max_retries = t
            .max_retries
            .or_else(|| self.llm.max_retries.filter(|_| is_default))
            .unwrap_or(def_retries);
        let max_tokens = t
            .max_tokens
            .or_else(|| self.llm.max_tokens.filter(|_| is_default))
            .or_else(|| anthropic.then_some(self.claude.max_tokens));
        let num_ctx = match p {
            Provider::Ollama => match t.num_ctx {
                Some(0) => None,
                Some(n) => Some(n),
                None => Some(OLLAMA_DEFAULT_NUM_CTX),
            },
            _ => None,
        };
        ProviderSettings {
            provider: p,
            base_url,
            api_key_env,
            request_timeout,
            max_retries,
            max_tokens,
            refusal_fallbacks: t.refusal_fallbacks.unwrap_or(self.claude.refusal_fallbacks),
            app_url: t.app_url.clone(),
            app_title: t.app_title.clone(),
            num_ctx,
        }
    }

    /// Provider, model and effort for a profile (`None`: the defaults).
    ///
    /// Model: the profile's, then `[llm.<provider>].model`, then `[llm].model`
    /// if the provider is the default, then (Anthropic) `[claude].model`.
    /// Providers other than Anthropic have no built-in model.
    pub fn profile_llm(&self, profile: Option<&ProfileConfig>) -> Result<LlmChoice> {
        let provider = profile
            .and_then(|p| p.provider)
            .unwrap_or_else(|| self.default_provider());
        let t = self.llm.table(provider);
        let is_default = provider == self.default_provider();
        let anthropic = provider == Provider::Anthropic;
        let model = profile
            .and_then(|p| p.model.clone())
            .or_else(|| t.model.clone())
            .or_else(|| self.llm.model.clone().filter(|_| is_default))
            .or_else(|| anthropic.then(|| self.claude.model.clone()))
            .filter(|m| !m.trim().is_empty());
        let Some(model) = model else {
            bail!(
                "no model for provider {provider:?}: set `model` in the profile,                  [llm.{provider}] or [llm] (only Anthropic has a default)"
            );
        };
        let effort = profile
            .and_then(|p| p.effort.clone())
            .or_else(|| t.effort.clone())
            .or_else(|| self.llm.effort.clone().filter(|_| is_default))
            .or_else(|| anthropic.then(|| self.claude.effort.clone()))
            .unwrap_or_default();
        Ok(LlmChoice {
            provider,
            model,
            effort,
        })
    }

    /// The approval mode for a profile: its own `approval_mode`, else
    /// `approval.mode`.
    pub fn approval_mode(&self, profile: Option<&ProfileConfig>) -> ApprovalMode {
        profile
            .and_then(|p| p.approval_mode)
            .unwrap_or(self.approval.mode)
    }

    /// Whether any profile (or the default) asks the classifier.
    pub fn uses_classifier(&self) -> bool {
        self.approval.mode != ApprovalMode::Manual
            || self
                .profiles
                .values()
                .any(|p| self.approval_mode(Some(p)) != ApprovalMode::Manual)
    }

    fn validate_approval(&self) -> Result<()> {
        let a = &self.approval;
        if self.uses_classifier() {
            let Some(u) = &a.base_url else {
                bail!("approval.base_url is required for shadow or classifier approval");
            };
            match url::Url::parse(u) {
                Ok(p) if p.scheme() == "http" || p.scheme() == "https" => {}
                _ => bail!(
                    "approval.base_url {:?} is not an http(s) URL",
                    display_url(u)
                ),
            }
        }
        if !(a.threshold > 0.0 && a.threshold < 1.0) {
            bail!("approval.threshold must be between 0 and 1 (exclusive)");
        }
        if let Some(b) = a.auto_approve_below
            && !(b > 0.0 && b <= a.threshold)
        {
            bail!("approval.auto_approve_below must be above 0 and at most approval.threshold");
        }
        if !(1..=MAX_APPROVAL_TIMEOUT_SECS).contains(&a.request_timeout_secs) {
            bail!(
                "approval.request_timeout_secs must be between 1 and {MAX_APPROVAL_TIMEOUT_SECS}"
            );
        }
        if a.alert_after_failures == 0 {
            bail!("approval.alert_after_failures must be at least 1");
        }
        if a.model.trim().is_empty() {
            bail!("approval.model must not be empty");
        }
        for (key, v) in [("instruct", &a.instruct), ("query", &a.query)] {
            if v.as_deref().is_some_and(|t| t.trim().is_empty()) {
                bail!("approval.{key} must not be empty (leave it out for the default)");
            }
        }
        Ok(())
    }

    /// Model calls per job: the profile's, `[llm]`, then `[claude]`.
    pub fn max_iterations(&self, profile: &ProfileConfig) -> u32 {
        profile
            .max_iterations
            .or(self.llm.max_iterations)
            .unwrap_or(self.claude.max_iterations)
    }

    /// Providers that jobs can start with: every profile's, or the default
    /// one when there are no profiles.
    pub fn used_providers(&self) -> std::collections::BTreeSet<Provider> {
        let mut used: std::collections::BTreeSet<Provider> = self
            .profiles
            .values()
            .map(|p| p.provider.unwrap_or_else(|| self.default_provider()))
            .collect();
        if used.is_empty() {
            used.insert(self.default_provider());
        }
        used
    }

    fn validate_llm(&self) -> Result<()> {
        let check = |p: Provider, e: &str, what: &str| -> Result<()> {
            if !p.efforts().contains(&e) {
                bail!(
                    "{what}: unknown effort {e:?} for {p} ({})",
                    p.efforts().join("|")
                );
            }
            Ok(())
        };
        if self.profiles.is_empty() {
            let c = self.profile_llm(None)?;
            if !c.effort.is_empty() {
                check(c.provider, &c.effort, "llm")?;
            }
        }
        for (name, profile) in &self.profiles {
            let c = self
                .profile_llm(Some(profile))
                .with_context(|| format!("profile {name}"))?;
            if !c.effort.is_empty() {
                check(c.provider, &c.effort, &format!("profile {name}"))?;
            }
        }
        for p in Provider::ALL {
            let t = self.llm.table(p);
            let s = self.provider_settings(p);
            match url::Url::parse(&s.base_url) {
                Ok(u) if u.scheme() == "http" || u.scheme() == "https" => {}
                _ => bail!(
                    "{p} base_url {:?} is not an http(s) URL",
                    display_url(&s.base_url)
                ),
            }
            if let Some(m) = &t.model
                && m.trim().is_empty()
            {
                bail!("[llm.{p}] model is empty");
            }
            if t.refusal_fallbacks.is_some() && p != Provider::Anthropic {
                bail!("[llm.{p}] refusal_fallbacks only applies to anthropic");
            }
            if (t.app_url.is_some() || t.app_title.is_some()) && p != Provider::OpenRouter {
                bail!("[llm.{p}] app_url/app_title only apply to openrouter");
            }
            if t.num_ctx.is_some() && p != Provider::Ollama {
                bail!("[llm.{p}] num_ctx only applies to ollama");
            }
            if let Some(u) = &t.app_url {
                url::Url::parse(u).with_context(|| format!("[llm.{p}] app_url {u:?}"))?;
                if reqwest::header::HeaderValue::from_str(u).is_err() {
                    bail!("[llm.{p}] app_url {u:?} is not a valid HTTP header value");
                }
            }
            if let Some(t) = &t.app_title
                && reqwest::header::HeaderValue::from_str(t).is_err()
            {
                bail!("[llm.{p}] app_title {t:?} must be printable ASCII (it is sent as a header)");
            }
        }
        Ok(())
    }

    /// Drop `[[mcp_servers]]` entries that cannot work, recording why. One
    /// bad entry must never stop the service or the other servers.
    fn prune_mcp_servers(&mut self) {
        let mut seen = std::collections::HashSet::new();
        let mut warnings = Vec::new();
        self.mcp_servers.retain(|s| {
            let problem = if s.name.is_empty()
                || !s
                    .name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            {
                Some("name must be lowercase letters, digits and '-'".to_string())
            } else if seen.contains(&s.name) {
                Some("duplicate name (the first entry is used)".to_string())
            } else {
                match url::Url::parse(&s.url) {
                    Ok(u) if u.scheme() == "http" || u.scheme() == "https" => None,
                    Ok(u) => Some(format!("url scheme {:?} is not http(s)", u.scheme())),
                    Err(e) => Some(format!("bad url {:?}: {e}", s.url)),
                }
            };
            match problem {
                Some(p) => {
                    warnings.push(format!("MCP server {:?} ignored: {p}", s.name));
                    false
                }
                None => {
                    seen.insert(s.name.clone());
                    true
                }
            }
        });
        self.mcp_warnings = warnings;
    }

    pub fn validate(&self) -> Result<()> {
        jiff::tz::TimeZone::get(&self.agent.timezone)
            .with_context(|| format!("unknown agent.timezone {:?}", self.agent.timezone))?;
        if !self.webhook.path.starts_with('/') {
            bail!("webhook.path must start with '/'");
        }
        if self.agent.attempt_timeout_secs == 0 {
            bail!("agent.attempt_timeout_secs must be at least 1");
        }
        if !(1..=MAX_WORKER_CONCURRENCY).contains(&self.agent.worker_concurrency) {
            bail!("agent.worker_concurrency must be between 1 and {MAX_WORKER_CONCURRENCY}");
        }
        if self.server.shutdown_grace_secs == 0 {
            bail!("server.shutdown_grace_secs must be at least 1");
        }
        if !(1..=MAX_DECODE_CONCURRENCY).contains(&self.stt.decode_concurrency) {
            bail!("stt.decode_concurrency must be between 1 and {MAX_DECODE_CONCURRENCY}");
        }
        if self.stt.decode_timeout_secs == 0 || self.stt.max_audio_secs == 0 {
            bail!("stt.decode_timeout_secs and stt.max_audio_secs must be at least 1");
        }
        let a = &self.alerts;
        if a.check_interval_secs == 0 || a.db_check_interval_secs == 0 {
            bail!(
                "alerts.check_interval_secs and alerts.db_check_interval_secs must be at least 1"
            );
        }
        if a.worker_stall_secs < 2 * WORKER_LOOP_MAX_WAIT_SECS {
            bail!(
                "alerts.worker_stall_secs must be at least {} (the worker loop runs every {WORKER_LOOP_MAX_WAIT_SECS} s)",
                2 * WORKER_LOOP_MAX_WAIT_SECS
            );
        }
        if a.db_error_threshold == 0 || a.failure_threshold == 0 || a.failure_window_secs == 0 {
            bail!(
                "alerts.db_error_threshold, alerts.failure_threshold and alerts.failure_window_secs must be at least 1"
            );
        }
        if a.ntfy_topic.as_deref().is_some_and(|t| t.trim().is_empty()) {
            bail!("alerts.ntfy_topic must not be empty (leave it out to use notify.ntfy.topic)");
        }
        self.validate_policies()?;
        let mut triggers = std::collections::HashMap::new();
        for (name, p) in &self.profiles {
            for t in &p.triggers {
                if let Some(other) = triggers.insert(t.clone(), name.clone()) {
                    bail!("trigger {t:?} is used by profiles {other:?} and {name:?}");
                }
            }
            for g in &p.tools {
                globset::Glob::new(g)
                    .with_context(|| format!("profile {name} tools glob {g:?}"))?;
            }
        }
        check_effort(&self.claude.effort)?;
        self.validate_llm()?;
        self.validate_approval()?;
        for u in [&self.stt.primary_url, &self.stt.fallback_url]
            .into_iter()
            .flatten()
        {
            url::Url::parse(u).with_context(|| format!("stt url {u:?}"))?;
        }
        if let Some(u) = &self.server.public_base_url {
            url::Url::parse(u).context("server.public_base_url")?;
        }
        Ok(())
    }

    fn validate_policies(&self) -> Result<()> {
        let a = &self.auth_codes;
        for (name, c) in &a.codes {
            if name.is_empty() || c.code_env.trim().is_empty() {
                bail!("auth_codes.codes entries need a name and a code_env");
            }
        }
        if !a.codes.is_empty() {
            if a.triggers
                .iter()
                .all(|t| crate::authcode::normalise(t).is_empty())
            {
                bail!("auth_codes.triggers must hold at least one phrase");
            }
            if a.valid_for_secs == 0
                || a.max_failures == 0
                || a.failure_window_secs == 0
                || a.lockout_secs == 0
            {
                bail!(
                    "auth_codes.valid_for_secs, max_failures, failure_window_secs and lockout_secs must be at least 1"
                );
            }
        }
        for (i, r) in self.tool_policies.iter().enumerate() {
            let at = || format!("tool_policies[{i}] (match {:?})", r.pattern);
            globset::Glob::new(&r.pattern).with_context(at)?;
            for (pointer, glob) in r.when.iter().chain(&r.when_any).chain(&r.unless) {
                if !pointer.starts_with('/') {
                    bail!(
                        "{}: condition {pointer:?} must be a JSON pointer starting with '/'",
                        at()
                    );
                }
                globset::Glob::new(glob)
                    .with_context(|| format!("{}: condition {pointer:?} glob", at()))?;
            }
            match (r.policy, &r.code) {
                (Policy::Code, Some(code)) if !a.codes.contains_key(code) => {
                    bail!("{}: code {code:?} is not in [auth_codes.codes]", at())
                }
                (Policy::Code, None) if a.codes.len() != 1 => bail!(
                    "{}: policy \"code\" needs code = \"<name>\" (one of [auth_codes.codes])",
                    at()
                ),
                (Policy::Code, _) => {}
                (_, Some(_)) => bail!("{}: code is only for policy \"code\"", at()),
                (_, None) if r.on_missing.is_some() => {
                    bail!("{}: on_missing is only for policy \"code\"", at())
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Checks that need the secrets too; run once at start-up.
    pub fn validate_runtime(&self, secrets: &Secrets) -> Result<()> {
        if self.webhook.require_signature && secrets.webhook_secret.is_none() {
            bail!(
                "webhook.require_signature is true but {ENV_WEBHOOK_SECRET} is not set \
                 (set it, or explicitly disable signature checks)"
            );
        }
        if let Some(s) = &secrets.webhook_secret
            && s.len() < 32
        {
            bail!("{ENV_WEBHOOK_SECRET} must be at least 32 characters");
        }
        // Only providers that a profile (or the default) uses need a key.
        for p in self.used_providers() {
            let settings = self.provider_settings(p);
            if p.needs_key() && !secrets.llm_keys.contains_key(&p) {
                bail!(
                    "{} is not set (needed for provider {p})",
                    settings
                        .api_key_env
                        .as_deref()
                        .unwrap_or(ENV_ANTHROPIC_API_KEY)
                );
            }
        }
        if self.notify.ntfy.is_some() && self.server.public_base_url.is_none() {
            tracing::warn!(
                "server.public_base_url is not set: confirmation notifications will have no Approve/Deny buttons"
            );
        }
        Ok(())
    }
}

fn check_effort(e: &str) -> Result<()> {
    match e {
        "low" | "medium" | "high" | "xhigh" | "max" => Ok(()),
        other => bail!("unknown effort {other:?} (low|medium|high|xhigh|max)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses() {
        let text = include_str!("../config.example.toml");
        let cfg = Config::from_toml(text).expect("example config is valid");
        assert_eq!(
            cfg.profile_for_trigger("single-click-hold"),
            Some("capture")
        );
        assert_eq!(
            cfg.profile_for_trigger("double-click-hold"),
            Some("command")
        );
        assert_eq!(cfg.profile_for_trigger("test-event"), None);
        assert!(cfg.stt.initial_prompt().unwrap().contains("Inbox"));
    }

    #[test]
    fn approval_defaults_to_manual_and_is_validated() {
        // Old configs have no [approval]: manual, no classifier needed.
        let cfg =
            Config::from_toml("[profiles.a]\ntriggers = [\"x\"]\nsystem_prompt = \"a\"\n").unwrap();
        assert_eq!(cfg.approval.mode, ApprovalMode::Manual);
        assert_eq!(
            cfg.approval_mode(cfg.profiles.get("a")),
            ApprovalMode::Manual
        );
        assert!(!cfg.uses_classifier());
        assert_eq!(cfg.approval.auto_approve_cutoff(), 0.5);
        assert!(
            cfg.approval
                .query()
                .contains("go beyond what the user asked for")
        );

        for bad in [
            "[approval]\nmode = \"classifier\"",
            "[approval]\nmode = \"shadow\"\nbase_url = \"ftp://m1.lan\"",
            "[approval]\nmode = \"automatic\"\nbase_url = \"http://m1.lan:8090\"",
            "[approval]\nbase_url = \"http://m1.lan:8090\"\nthreshold = 1.0",
            "[approval]\nbase_url = \"http://m1.lan:8090\"\nauto_approve_below = 0.6",
            "[approval]\nbase_url = \"http://m1.lan:8090\"\nauto_approve_below = 0.0",
            "[approval]\nbase_url = \"http://m1.lan:8090\"\nrequest_timeout_secs = 0",
            "[approval]\nbase_url = \"http://m1.lan:8090\"\nquery = \" \"",
            "[approval]\nbase_url = \"http://m1.lan:8090\"\nunknown = 1",
            // A profile asking for the classifier needs the server too.
            "[profiles.a]\ntriggers = [\"x\"]\nsystem_prompt = \"a\"\napproval_mode = \"classifier\"",
        ] {
            assert!(Config::from_toml(bad).is_err(), "{bad}");
        }

        let cfg = Config::from_toml(
            r#"
            [approval]
            mode = "classifier"
            base_url = "http://m1.lan:8090"
            auto_approve_below = 0.2
            query = "Is this unsafe?"
            [profiles.a]
            triggers = ["x"]
            system_prompt = "a"
            [profiles.b]
            triggers = ["y"]
            system_prompt = "b"
            approval_mode = "manual"
            "#,
        )
        .unwrap();
        assert!(cfg.uses_classifier());
        assert_eq!(
            cfg.approval_mode(cfg.profiles.get("a")),
            ApprovalMode::Classifier
        );
        assert_eq!(
            cfg.approval_mode(cfg.profiles.get("b")),
            ApprovalMode::Manual
        );
        assert_eq!(cfg.approval.auto_approve_cutoff(), 0.2);
        assert_eq!(cfg.approval.query(), "Is this unsafe?");
        assert_eq!(cfg.approval.instruct(), DEFAULT_APPROVAL_INSTRUCT);

        // Only one profile on shadow is enough to need the server.
        let cfg = Config::from_toml(
            "[approval]\nbase_url = \"http://m1.lan:8090\"\n[profiles.a]\ntriggers = [\"x\"]\nsystem_prompt = \"a\"\napproval_mode = \"shadow\"\n",
        )
        .unwrap();
        assert!(cfg.uses_classifier());
    }

    #[test]
    fn duplicate_trigger_rejected() {
        let text = r#"
            [profiles.a]
            triggers = ["single-click-hold"]
            system_prompt = "a"
            [profiles.b]
            triggers = ["single-click-hold"]
            system_prompt = "b"
        "#;
        assert!(Config::from_toml(text).is_err());
    }

    #[test]
    fn worker_limits_have_safe_defaults_and_are_checked() {
        let cfg = Config::from_toml("").unwrap();
        assert_eq!(cfg.agent.worker_concurrency, 2);
        assert_eq!(cfg.agent.attempt_timeout_secs, 900);
        for bad in [
            "[agent]\nworker_concurrency = 0",
            "[agent]\nworker_concurrency = 17",
            "[agent]\nattempt_timeout_secs = 0",
        ] {
            assert!(Config::from_toml(bad).is_err(), "{bad}");
        }
        let cfg = Config::from_toml("[agent]\nworker_concurrency = 1\nattempt_timeout_secs = 60")
            .unwrap();
        assert_eq!(cfg.agent.worker_concurrency, 1);
    }

    #[test]
    fn alert_and_shutdown_settings_are_checked() {
        let cfg = Config::from_toml("").unwrap();
        assert!(cfg.alerts.enabled);
        assert_eq!(cfg.alerts.ntfy_topic, None);
        assert_eq!(cfg.alerts.worker_stall_secs, 120);
        assert_eq!(cfg.server.shutdown_grace_secs, 8);
        assert_eq!(cfg.stt.decode_concurrency, 1);
        for bad in [
            "[alerts]\nworker_stall_secs = 30",
            "[alerts]\ncheck_interval_secs = 0",
            "[alerts]\nfailure_threshold = 0",
            "[alerts]\nntfy_topic = \" \"",
            "[alerts]\nunknown = 1",
            "[server]\nshutdown_grace_secs = 0",
            "[stt]\ndecode_concurrency = 0",
            "[stt]\ndecode_concurrency = 9",
            "[stt]\ndecode_timeout_secs = 0",
        ] {
            assert!(Config::from_toml(bad).is_err(), "{bad}");
        }
        let cfg = Config::from_toml("[alerts]\nenabled = false\nntfy_topic = \"ops\"").unwrap();
        assert!(!cfg.alerts.enabled);
        assert_eq!(cfg.alerts.ntfy_topic.as_deref(), Some("ops"));
    }

    #[test]
    fn fallbacks_auto_skips_haiku() {
        let c = ClaudeConfig::default();
        assert!(c.fallbacks_enabled("claude-opus-5-5"));
        assert!(c.fallbacks_enabled("claude-sonnet-5-5"));
        assert!(!c.fallbacks_enabled("claude-haiku-5-5"));
    }

    #[test]
    fn bad_mcp_servers_are_dropped_not_fatal() {
        let cfg = Config::from_toml(
            r#"
[[mcp_servers]]
name = "deck"
url = "http://nextcloud-mcp:8000/mcp"

[[mcp_servers]]
name = "Bad Name"
url = "http://x/mcp"

[[mcp_servers]]
name = "deck"
url = "http://other/mcp"

[[mcp_servers]]
name = "nourl"
url = "not a url"

[[mcp_servers]]
name = "ftp"
url = "ftp://x/mcp"
"#,
        )
        .expect("bad MCP entries must not make the config invalid");
        let names: Vec<_> = cfg.mcp_servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["deck"]);
        assert_eq!(cfg.mcp_servers[0].url, "http://nextcloud-mcp:8000/mcp");
        assert_eq!(cfg.mcp_warnings.len(), 4, "{:?}", cfg.mcp_warnings);
    }

    #[test]
    fn mcp_basic_credentials_are_standard_padded_base64() {
        // Expected values from coreutils `base64`: no padding, one `=`, two
        // `=`, and a UTF-8 password containing colons (RFC 7617 splits on
        // the first colon only).
        for (user, pw, want) in [
            ("abcd", "e", "YWJjZDpl"),
            ("ab", "cd", "YWI6Y2Q="),
            ("ab", "c", "YWI6Yw=="),
            (
                "rob",
                "pässwörd:with:colons",
                "cm9iOnDDpHNzd8O2cmQ6d2l0aDpjb2xvbnM=",
            ),
        ] {
            let servers: Vec<McpServerConfig> = toml::from_str::<Config>(&format!(
                "[[mcp_servers]]\nname = \"nc\"\nurl = \"http://nc/mcp\"\n\
                 basic_auth = {{ username = \"{user}\", password_env = \"PW\" }}\n"
            ))
            .unwrap()
            .mcp_servers;
            let (h, warnings) = mcp_headers(&servers, |_| Ok(Some(pw.to_string())));
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(
                h["nc"],
                [("Authorization".to_string(), format!("Basic {want}"))],
                "{user}:{pw}"
            );
        }
    }

    #[test]
    fn mcp_headers_basic_bearer_and_custom() {
        let servers: Vec<McpServerConfig> = toml::from_str::<Config>(
            r#"
[[mcp_servers]]
name = "nc"
url = "https://cloud.example/apps/mcp/"
token_env = "IGNORED"
basic_auth = { username = "rob", password_env = "NC_PW" }
headers_env = { "X-Extra" = "EXTRA", "Accept" = "EXTRA", "Authorization" = "EXTRA" }

[[mcp_servers]]
name = "tok"
url = "http://tok/mcp"
token_env = "TOK"

[[mcp_servers]]
name = "missing"
url = "http://missing/mcp"
basic_auth = { username = "x", password_env = "UNSET" }
"#,
        )
        .unwrap()
        .mcp_servers;
        let env: HashMap<&str, &str> = [
            ("NC_PW", "app-pass"),
            ("TOK", "t0k"),
            ("EXTRA", "v"),
            ("IGNORED", "no"),
        ]
        .into();
        let (h, warnings) = mcp_headers(&servers, |k| Ok(env.get(k).map(|v| v.to_string())));

        let nc = &h["nc"];
        assert!(
            nc.contains(&("Authorization".into(), "Basic cm9iOmFwcC1wYXNz".into())),
            "{nc:?}"
        );
        assert!(nc.contains(&("X-Extra".into(), "v".into())));
        assert_eq!(
            nc.len(),
            2,
            "Accept and a second Authorization are refused: {nc:?}"
        );
        assert_eq!(
            h["tok"],
            vec![("Authorization".to_string(), "Bearer t0k".to_string())]
        );
        assert!(!h.contains_key("missing"), "tried without credentials");
        // token_env ignored, Accept reserved, Authorization duplicate, UNSET missing.
        assert_eq!(warnings.len(), 4, "{warnings:?}");
    }

    #[test]
    fn bad_effort_rejected() {
        assert!(Config::from_toml("[claude]\neffort = \"turbo\"\n").is_err());
        // Valid for Claude, not for Mistral.
        let e = Config::from_toml(
            "[llm]\nprovider = \"mistral\"\nmodel = \"mistral-small-latest\"\neffort = \"turbo\"\n",
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("effort"), "{e:#}");
    }

    const PROFILE: &str = "[profiles.p]\ntriggers = [\"t\"]\nsystem_prompt = \"x\"\n";

    #[test]
    fn legacy_claude_section_maps_to_anthropic() {
        let cfg = Config::from_toml(&format!(
            "[claude]\napi_base = \"https://proxy.example\"\nmodel = \"claude-haiku-5-5\"\neffort = \"medium\"\n\
             max_tokens = 4000\nmax_iterations = 3\nrefusal_fallbacks = \"off\"\nrequest_timeout_secs = 30\nmax_retries = 2\n{PROFILE}"
        ))
        .unwrap();
        let p = &cfg.profiles["p"];
        assert_eq!(
            cfg.profile_llm(Some(p)).unwrap(),
            LlmChoice {
                provider: Provider::Anthropic,
                model: "claude-haiku-5-5".into(),
                effort: "medium".into()
            }
        );
        assert_eq!(cfg.max_iterations(p), 3);
        let s = cfg.provider_settings(Provider::Anthropic);
        assert_eq!(s.base_url, "https://proxy.example");
        assert_eq!(s.max_tokens, Some(4000));
        assert_eq!(s.refusal_fallbacks, FallbackMode::Off);
        assert_eq!(s.request_timeout.as_secs(), 30);
        assert_eq!(s.max_retries, 2);
        assert_eq!(s.api_key_env.as_deref(), Some("ANTHROPIC_API_KEY"));
        assert_eq!(
            cfg.used_providers().into_iter().collect::<Vec<_>>(),
            [Provider::Anthropic]
        );

        // No config at all: Claude Opus 5.5 at low effort, as before.
        let cfg = Config::from_toml("").unwrap();
        let c = cfg.profile_llm(None).unwrap();
        assert_eq!(
            (c.provider, c.model.as_str(), c.effort.as_str()),
            (Provider::Anthropic, "claude-opus-5-5", "low")
        );
        assert_eq!(
            cfg.provider_settings(Provider::Anthropic).max_tokens,
            Some(16000)
        );
    }

    #[test]
    fn llm_section_wins_over_claude() {
        let cfg = Config::from_toml(
            "[claude]\nmodel = \"claude-haiku-5-5\"\nmax_iterations = 3\n\
             [llm]\nmodel = \"claude-sonnet-5-5\"\nmax_iterations = 5\nbase_url = \"https://a.example\"\n",
        )
        .unwrap();
        assert_eq!(cfg.profile_llm(None).unwrap().model, "claude-sonnet-5-5");
        assert_eq!(
            cfg.provider_settings(Provider::Anthropic).base_url,
            "https://a.example"
        );
        assert_eq!(cfg.llm.max_iterations, Some(5));
    }

    #[test]
    fn unknown_provider_rejected() {
        let e = Config::from_toml("[llm]\nprovider = \"gemini\"\n").unwrap_err();
        assert!(format!("{e:#}").contains("unknown variant"), "{e:#}");
        let e = Config::from_toml(&format!("{PROFILE}provider = \"chatgpt\"\n")).unwrap_err();
        assert!(format!("{e:#}").contains("unknown variant"), "{e:#}");
    }

    #[test]
    fn model_required_for_providers_without_default() {
        for p in ["openai", "mistral", "openrouter", "ollama"] {
            let e = Config::from_toml(&format!("[llm]\nprovider = \"{p}\"\n")).unwrap_err();
            assert!(format!("{e:#}").contains("no model"), "{p}: {e:#}");
            let e = Config::from_toml(&format!("{PROFILE}provider = \"{p}\"\n")).unwrap_err();
            assert!(format!("{e:#}").contains("profile p"), "{p}: {e:#}");
        }
        // A Claude model under [llm] does not leak into another provider.
        let e = Config::from_toml(&format!(
            "[llm]\nmodel = \"claude-sonnet-5-5\"\n{PROFILE}provider = \"ollama\"\n"
        ))
        .unwrap_err();
        assert!(format!("{e:#}").contains("no model"), "{e:#}");
        // The provider table's model is used.
        let cfg = Config::from_toml(&format!(
            "[llm.ollama]\nmodel = \"qwen3:8b\"\n{PROFILE}provider = \"ollama\"\n"
        ))
        .unwrap();
        assert_eq!(
            cfg.profile_llm(Some(&cfg.profiles["p"])).unwrap().model,
            "qwen3:8b"
        );
    }

    #[test]
    fn per_profile_provider_and_settings() {
        let cfg = Config::from_toml(&format!(
            "[llm]\nprovider = \"anthropic\"\nrequest_timeout_secs = 60\n\
             [llm.ollama]\nbase_url = \"http://m1.lan:11434\"\nnum_ctx = 16384\n\
             {PROFILE}provider = \"ollama\"\nmodel = \"qwen3:8b\"\neffort = \"high\"\n\
             [profiles.q]\ntriggers = [\"u\"]\nsystem_prompt = \"y\"\n"
        ))
        .unwrap();
        let c = cfg.profile_llm(Some(&cfg.profiles["p"])).unwrap();
        assert_eq!((c.provider, c.effort.as_str()), (Provider::Ollama, "high"));
        let c = cfg.profile_llm(Some(&cfg.profiles["q"])).unwrap();
        assert_eq!(
            (c.provider, c.effort.as_str()),
            (Provider::Anthropic, "low")
        );
        let o = cfg.provider_settings(Provider::Ollama);
        assert_eq!(o.base_url, "http://m1.lan:11434");
        assert_eq!(o.num_ctx, Some(16384));
        assert_eq!(o.api_key_env, None);
        // [llm] timeout is for the default provider; Ollama keeps its own.
        assert_eq!(o.request_timeout.as_secs(), 600);
        assert_eq!(
            cfg.provider_settings(Provider::Anthropic)
                .request_timeout
                .as_secs(),
            60
        );
        assert_eq!(
            cfg.used_providers().into_iter().collect::<Vec<_>>(),
            [Provider::Anthropic, Provider::Ollama]
        );
        // Unused providers keep built-in defaults.
        let m = cfg.provider_settings(Provider::Mistral);
        assert_eq!(m.base_url, "https://api.mistral.ai");
        assert_eq!(m.api_key_env.as_deref(), Some("MISTRAL_API_KEY"));
        assert_eq!(m.max_tokens, None);
    }

    #[test]
    fn credentials_are_redacted_from_text() {
        let text =
            "echo: Authorization: Bearer tok-123456; X-Api-Key: key-abcdef; Basic dXNlcjpwYXNz";
        assert_eq!(
            redact_credentials(
                text,
                [
                    "Bearer tok-123456",
                    "key-abcdef",
                    "Basic dXNlcjpwYXNz",
                    "ab",
                    ""
                ]
            ),
            "echo: Authorization: [redacted]; X-Api-Key: [redacted]; [redacted]"
        );
        assert_eq!(redact_credentials("Bearer x", None::<&str>), "Bearer x");
    }

    #[test]
    fn display_url_hides_passwords() {
        assert_eq!(
            display_url("https://ring:s3cret-pw@ollama.example/"),
            "https://ring:***@ollama.example/"
        );
        assert_eq!(display_url("http://m1.lan:11434"), "http://m1.lan:11434");
        assert_eq!(display_url("not a url"), "not a url");
        let e =
            Config::from_toml("[llm.ollama]\nbase_url = \"ftp://u:s3cret-pw@host\"\n").unwrap_err();
        assert!(!format!("{e:#}").contains("s3cret-pw"), "{e:#}");
    }

    #[test]
    fn attribution_headers_must_be_valid_header_values() {
        assert!(Config::from_toml("[llm.openrouter]\napp_title = \"Ringmaster\"\n").is_ok());
        assert!(Config::from_toml("[llm.openrouter]\napp_title = \"Ring\\nmaster\"\n").is_err());
    }

    #[test]
    fn provider_specific_keys_checked() {
        assert!(Config::from_toml("[llm.openai]\nnum_ctx = 4096\n").is_err());
        assert!(Config::from_toml("[llm.ollama]\napp_title = \"x\"\n").is_err());
        assert!(Config::from_toml("[llm.mistral]\nrefusal_fallbacks = \"on\"\n").is_err());
        assert!(Config::from_toml("[llm.ollama]\nbase_url = \"m1.lan:11434\"\n").is_err());
        assert!(Config::from_toml("[llm.ollama]\nnum_ctx = 0\n").is_ok());
    }

    #[test]
    fn keys_read_only_matter_for_used_providers() {
        let cfg = Config::from_toml(&format!(
            "[llm]\nprovider = \"ollama\"\nmodel = \"qwen3:8b\"\n{PROFILE}"
        ))
        .unwrap();
        let secrets = Secrets {
            webhook_secret: Some("x".repeat(32)),
            ..Default::default()
        };
        cfg.validate_runtime(&secrets).expect("Ollama needs no key");
        let cfg = Config::from_toml(&format!(
            "[llm]\nprovider = \"openrouter\"\nmodel = \"openai/gpt-6.1-sol\"\napi_key_env = \"OR_KEY\"\n{PROFILE}"
        ))
        .unwrap();
        let e = cfg.validate_runtime(&secrets).unwrap_err().to_string();
        assert!(e.contains("OR_KEY"), "{e}");
    }

    #[test]
    fn argument_conditions_and_code_rules_are_validated() {
        let base = "[auth_codes.codes.alarm]\ncode_env = \"RINGMASTER_CODE_ALARM\"\n";
        let ok = |extra: &str| Config::from_toml(&format!("{base}{extra}"));
        let rule = |body: &str| format!("[[tool_policies]]\nmatch = \"ha/*\"\n{body}\n");
        // A good one: conditions, a code rule, the only code left out.
        let c = ok(&format!(
            "{}{}",
            rule("when = { \"/action\" = \"alarm_control_panel.*\" }\nunless = { \"/target/entity_id\" = \"x\" }\npolicy = \"deny\""),
            rule("policy = \"code\"\non_missing = \"ask\"")
        ))
        .unwrap();
        assert!(c.tool_policies[0].is_conditional());
        assert_eq!(c.tool_policies[1].on_missing, Some(OnMissing::Ask));
        assert_eq!(c.auth_codes.valid_for_secs, 300);
        assert_eq!(c.auth_codes.triggers.len(), 4);
        for (bad, why) in [
            (
                rule("policy = \"code\"\ncode = \"garage\""),
                "not in [auth_codes.codes]",
            ),
            (
                rule("policy = \"allow\"\ncode = \"alarm\""),
                "only for policy",
            ),
            (
                rule("policy = \"allow\"\non_missing = \"deny\""),
                "only for policy",
            ),
            (
                rule("when = { \"action\" = \"x\" }\npolicy = \"deny\""),
                "JSON pointer",
            ),
            (rule("when = { \"/a\" = \"[\" }\npolicy = \"deny\""), "glob"),
            (
                rule("policy = \"code\"\nwhen = { \"/a\" = \"x\" }\nbogus = 1"),
                "unknown field",
            ),
        ] {
            let e = format!("{:#}", ok(&bad).unwrap_err());
            assert!(e.contains(why), "{bad}: {e}");
        }
        // Two codes: a code rule must name one.
        let two = format!(
            "{base}[auth_codes.codes.garage]\ncode_env = \"G\"\n{}",
            rule("policy = \"code\"")
        );
        assert!(format!("{:#}", Config::from_toml(&two).unwrap_err()).contains("needs code"));
        // No codes at all: nothing to name.
        let none = Config::from_toml(&rule("policy = \"code\"")).unwrap_err();
        assert!(format!("{none:#}").contains("needs code"), "{none:#}");
        // Code values never go in the file.
        let e = Config::from_toml("[auth_codes.codes.alarm]\ncode = \"delta\"\n").unwrap_err();
        assert!(format!("{e:#}").contains("unknown field"), "{e:#}");
    }

    #[test]
    fn registered_secrets_are_redacted_everywhere() {
        register_secret("kilo lima 9");
        assert_eq!(
            redact_credentials("echoed: kilo lima 9!", None::<&str>),
            "echoed: [redacted]!"
        );
        // Only the whole phrase, not its parts.
        assert_eq!(redact_credentials("lima 9", None::<&str>), "lima 9");
    }
}
