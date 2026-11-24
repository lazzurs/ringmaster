//! Configuration: a TOML file for everything that is not secret, environment
//! variables for everything that is.
//!
//! Every secret can also be given as `<NAME>_FILE` pointing at a file holding
//! the value (Docker / Compose secrets).

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

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
    #[serde(default)]
    pub claude: ClaudeConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub retention: RetentionConfig,
    #[serde(default)]
    pub notify: NotifyConfig,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    #[serde(default)]
    pub tool_policies: Vec<ToolPolicyRule>,
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".parse().expect("static address"),
            public_base_url: None,
            data_dir: PathBuf::from("/data"),
            max_body_bytes: 25 * 1024 * 1024,
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

impl ClaudeConfig {
    pub fn fallbacks_enabled(&self, model: &str) -> bool {
        match self.refusal_fallbacks {
            FallbackMode::On => true,
            FallbackMode::Off => false,
            FallbackMode::Auto => !model.starts_with("claude-haiku"),
        }
    }
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
        }
    }
}

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

#[derive(Debug, Clone, Deserialize)]
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
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPolicyRule {
    /// Glob over `<server>/<tool>`, e.g. `homey/*lock*`.
    #[serde(rename = "match")]
    pub pattern: String,
    pub policy: Policy,
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
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub job_max_age_secs: Option<u64>,
}

/// Secrets read from the environment.
#[derive(Clone, Default)]
pub struct Secrets {
    pub anthropic_api_key: Option<String>,
    pub webhook_secret: Option<String>,
    pub ntfy_token: Option<String>,
    /// MCP server name -> bearer token.
    pub mcp_tokens: HashMap<String, String>,
    /// MCP token problems; the server is still tried without a token.
    pub warnings: Vec<String>,
}

impl std::fmt::Debug for Secrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Secrets")
            .field("anthropic_api_key", &self.anthropic_api_key.is_some())
            .field("webhook_secret", &self.webhook_secret.is_some())
            .field("ntfy_token", &self.ntfy_token.is_some())
            .field("mcp_tokens", &self.mcp_tokens.keys().collect::<Vec<_>>())
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

impl Secrets {
    pub fn from_env(config: &Config) -> Result<Self> {
        let mut mcp_tokens = HashMap::new();
        let mut warnings = Vec::new();
        for s in &config.mcp_servers {
            let Some(var) = &s.token_env else { continue };
            match secret_from_env(var) {
                Ok(Some(tok)) => {
                    mcp_tokens.insert(s.name.clone(), tok);
                }
                Ok(None) => warnings.push(format!(
                    "MCP server {:?}: {var} is not set; connecting without a token",
                    s.name
                )),
                Err(e) => warnings.push(format!(
                    "MCP server {:?}: {e:#}; connecting without a token",
                    s.name
                )),
            }
        }
        Ok(Self {
            anthropic_api_key: secret_from_env(ENV_ANTHROPIC_API_KEY)?,
            webhook_secret: secret_from_env(ENV_WEBHOOK_SECRET)?,
            ntfy_token: secret_from_env(ENV_NTFY_TOKEN)?,
            mcp_tokens,
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
        for r in &self.tool_policies {
            globset::Glob::new(&r.pattern)
                .with_context(|| format!("tool_policies match {:?}", r.pattern))?;
        }
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
            if let Some(e) = &p.effort {
                check_effort(e)?;
            }
        }
        check_effort(&self.claude.effort)?;
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
        if secrets.anthropic_api_key.is_none() {
            bail!("{ENV_ANTHROPIC_API_KEY} is not set");
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
    fn bad_effort_rejected() {
        assert!(Config::from_toml("[claude]\neffort = \"turbo\"\n").is_err());
    }
}
