//! `ringmaster eval`: compare models on synthetic voice commands, through
//! the same request code as production.
//!
//! Every request goes through the real provider adapters ([`Llms`]: request
//! bodies, `num_ctx`, effort, retries, redaction), with the production
//! system prompt ([`agent::system_prompt`]), first message
//! ([`agent::context_text`]) and tool list ([`agent::prepare_tools`], so
//! schema checks, profile filters and sorting apply). There are no MCP
//! servers: tools come from fixture files and their results are scripted.
//! The approval classifier suite uses [`Classifier::classify`] itself.
//!
//! Suites (see `docs/model-evaluation.md`):
//! - `agent`: the first action a model takes for one request;
//! - `capture`: the whole capture loop against a scripted Deck;
//! - `judge`: a model asked to approve or hold proposed calls;
//! - `classifier`: the configured Shieldstral classifier on the same calls.

pub mod fixtures;
pub mod judge;
pub mod report;
pub mod stats;
pub mod suites;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent;
use crate::classifier::Classifier;
use crate::config::{Config, ProfileConfig};
use crate::llm::{Llms, Provider};

use fixtures::Fixtures;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Suite {
    Agent,
    Capture,
    Judge,
    Classifier,
}

impl Suite {
    pub const ALL: [Suite; 4] = [
        Suite::Agent,
        Suite::Capture,
        Suite::Judge,
        Suite::Classifier,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Suite::Agent => "agent",
            Suite::Capture => "capture",
            Suite::Judge => "judge",
            Suite::Classifier => "classifier",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|x| x.as_str() == s.trim().to_ascii_lowercase())
            .with_context(|| format!("unknown suite {s:?} (agent, capture, judge, classifier)"))
    }
}

/// What `ringmaster eval` was asked to do.
#[derive(Debug, Clone)]
pub struct Options {
    pub config: PathBuf,
    pub cases: PathBuf,
    pub suites: Vec<Suite>,
    /// `provider[:model]`; empty: the config's default model.
    pub models: Vec<String>,
    /// Distractor counts to run with (each one is a separate pass).
    pub distractors: Vec<usize>,
    pub repeat: u32,
    pub output: PathBuf,
    pub report: PathBuf,
}

/// A model to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub provider: Provider,
    pub model: String,
    /// Resolved from the config as for a profile using this provider.
    pub effort: String,
}

impl ModelSpec {
    /// `provider:model`, as in results and the report.
    pub fn label(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }

    /// `provider[:model]`; the model may contain `:` (`ollama:qwen3:8b`).
    /// Without a model, the config's model for that provider is used.
    pub fn parse(spec: &str, config: &Config) -> Result<Self> {
        let (p, m) = spec.split_once(':').unwrap_or((spec, ""));
        let provider: Provider = serde_json::from_value(json!(p.trim().to_ascii_lowercase()))
            .map_err(|_| {
                anyhow::anyhow!(
                    "--model {spec:?}: unknown provider {p:?} (anthropic, openai, mistral, openrouter, ollama)"
                )
            })?;
        let profile = ProfileConfig {
            triggers: Vec::new(),
            system_prompt: String::new(),
            tools: Vec::new(),
            provider: Some(provider),
            model: Some(m.trim().to_string()).filter(|m| !m.is_empty()),
            effort: None,
            max_iterations: None,
            job_max_age_secs: None,
            approval_mode: None,
            require: None,
        };
        let choice = config
            .profile_llm(Some(&profile))
            .with_context(|| format!("--model {spec:?}"))?;
        Ok(Self {
            provider: choice.provider,
            model: choice.model,
            effort: choice.effort,
        })
    }

    /// The config's default model (`[llm]`).
    pub fn default_for(config: &Config) -> Result<Self> {
        let choice = config.profile_llm(None)?;
        Ok(Self {
            provider: choice.provider,
            model: choice.model,
            effort: choice.effort,
        })
    }
}

/// Token counts and durations a provider reported for one request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Every prompt token, cached or not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Prompt tokens served from the provider's cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Prompt tokens written to the cache (Anthropic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Ollama: loading the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_ms: Option<f64>,
    /// Ollama: reading the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_ms: Option<f64>,
    /// Ollama: generating the reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_ms: Option<f64>,
}

impl Usage {
    /// From a turn's audit summary (`usage`), whatever the provider.
    pub fn from_audit(audit: &Value) -> Self {
        let Some(u) = audit.get("usage").filter(|u| u.is_object()) else {
            return Self::default();
        };
        let n = |k: &str| u.get(k).and_then(Value::as_u64);
        let f = |k: &str| u.get(k).and_then(Value::as_f64);
        if u.get("cache_read_input_tokens").is_some()
            || u.get("cache_creation_input_tokens").is_some()
        {
            // Anthropic: input_tokens leaves out cached and cache-written tokens.
            let (input, read, write) = (
                n("input_tokens"),
                n("cache_read_input_tokens"),
                n("cache_creation_input_tokens"),
            );
            let prompt = (input.is_some() || read.is_some() || write.is_some())
                .then(|| input.unwrap_or(0) + read.unwrap_or(0) + write.unwrap_or(0));
            return Self {
                prompt_tokens: prompt,
                output_tokens: n("output_tokens"),
                cached_tokens: read,
                cache_write_tokens: write,
                ..Default::default()
            };
        }
        Self {
            prompt_tokens: n("input_tokens"),
            output_tokens: n("output_tokens"),
            cached_tokens: n("cached_tokens"),
            cache_write_tokens: None,
            load_ms: f("load_ms"),
            prompt_eval_ms: f("prompt_eval_ms"),
            eval_ms: f("eval_ms"),
        }
    }
}

/// One tool call in a reply.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Call {
    /// `<server>/<tool>`, or the name the model used for a tool that was
    /// not offered.
    pub tool: String,
    /// The tool was offered.
    pub known: bool,
    /// The arguments (`null` when they were not a JSON object).
    pub arguments: Value,
    /// Why the arguments are unusable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid: Option<String>,
    /// Agent suite: what the call counts as.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
    /// Whether the scripted result was an error (unset: not answered).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_error: Option<bool>,
    /// The scripted result's text, for the capture profile's `require`
    /// (not written to `results.jsonl`).
    #[serde(skip)]
    pub result_text: Option<String>,
}

/// One request, as written to `results.jsonl`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub suite: String,
    pub case: String,
    pub category: String,
    pub model: String,
    pub distractors: usize,
    /// 1-based run of the case (`--repeat`).
    pub repeat: u32,
    /// 1-based request within the case.
    pub turn: u32,
    /// The case's last request: its outcome is final.
    pub last: bool,
    /// The case's outcome (the same on every row of the case).
    pub outcome: String,
    pub pass: bool,
    /// `harmful`, `unknown_tool`, `invalid_args`, `false_success`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<Call>,
    /// The reply's text (shortened).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// An error, or how the outcome was reached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Judge and classifier: `safe` or `unsafe`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Judge: `run` or `ask`; classifier: `safe`, `uncertain` or `unsafe`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_mismatch: Option<f64>,
    /// Classifier score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub tools_offered: usize,
    /// First request of this model with this system prompt and tool list.
    pub cold: bool,
    pub wall_ms: f64,
    #[serde(default)]
    pub usage: Usage,
}

/// Facts about the run for the report header.
#[derive(Debug, Clone, Default)]
pub struct Meta {
    pub version: String,
    pub started: String,
    pub suites: Vec<Suite>,
    pub models: Vec<String>,
    pub distractors: Vec<usize>,
    pub repeat: u32,
    pub classifier: Option<ClassifierMeta>,
    /// Suites or parts that did not run, and why.
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ClassifierMeta {
    pub label: String,
    pub threshold: f64,
    pub cutoff: f64,
}

/// How a run went.
#[derive(Debug, Clone)]
pub struct Summary {
    pub rows: usize,
    /// Harmful agent calls, unsafe calls a judge would run, and unsafe calls
    /// the classifier would approve at the configured cut-off.
    pub harmful: usize,
}

/// Writes rows as they come, with every secret and endpoint removed.
pub(crate) struct Sink {
    file: std::io::BufWriter<std::fs::File>,
    secrets: Vec<String>,
    /// Base URLs and their `host:port`, longest first.
    endpoints: Vec<String>,
    pub rows: Vec<Row>,
}

/// The spellings of an endpoint that error messages may contain: the base
/// URL as configured, its origin, and its `host:port`.
fn endpoint_forms(base: &str) -> Vec<String> {
    let base = base.trim().trim_end_matches('/');
    let mut forms = vec![base.to_string()];
    if let Ok(u) = url::Url::parse(base) {
        forms.push(u.origin().ascii_serialization());
        if let (Some(host), Some(port)) = (u.host_str(), u.port_or_known_default()) {
            forms.push(format!("{host}:{port}"));
        }
    }
    forms.retain(|f| f.len() >= 4);
    forms
}

impl Sink {
    fn create(path: &std::path::Path, secrets: Vec<String>, endpoints: &[String]) -> Result<Self> {
        let file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut endpoints: Vec<String> = endpoints.iter().flat_map(|e| endpoint_forms(e)).collect();
        endpoints.sort_by_key(|e| std::cmp::Reverse(e.len()));
        endpoints.dedup();
        Ok(Self {
            file: std::io::BufWriter::new(file),
            secrets,
            endpoints,
            rows: Vec::new(),
        })
    }

    pub(crate) fn redact(&self, text: &str) -> String {
        let mut out =
            crate::config::redact_credentials(text, self.secrets.iter().map(String::as_str));
        for e in &self.endpoints {
            out = out.replace(e.as_str(), "[endpoint]");
        }
        out
    }

    pub(crate) fn push(&mut self, row: Row) -> Result<()> {
        let line = self.redact(&serde_json::to_string(&row)?);
        // Keep the in-memory copy identical to what was written.
        let row: Row = serde_json::from_str(&line)?;
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        self.rows.push(row);
        Ok(())
    }
}

/// Run the eval. `secret` reads an environment variable (or its `_FILE`),
/// as [`crate::config::secret_from_env`] does in production.
pub async fn run(
    opts: &Options,
    secret: &dyn Fn(&str) -> Result<Option<String>>,
) -> Result<Summary> {
    let config = Config::load(&opts.config)?;
    let fixtures = Fixtures::load(&opts.cases)?;
    if opts.repeat == 0 {
        bail!("--repeat must be at least 1");
    }
    let suites: BTreeSet<Suite> = opts.suites.iter().copied().collect();
    let distractors = if opts.distractors.is_empty() {
        vec![0]
    } else {
        opts.distractors.clone()
    };
    for n in &distractors {
        fixtures.distractors(*n)?;
    }
    let models: Vec<ModelSpec> = if opts.models.is_empty() {
        vec![ModelSpec::default_for(&config)?]
    } else {
        opts.models
            .iter()
            .map(|m| ModelSpec::parse(m, &config))
            .collect::<Result<_>>()?
    };
    let needs_models = suites
        .iter()
        .any(|s| matches!(s, Suite::Agent | Suite::Capture | Suite::Judge));

    // Keys exactly as production finds them: only the providers in use.
    let mut keys = BTreeMap::new();
    let mut secrets = Vec::new();
    let mut endpoints: Vec<String> = config.approval.base_url.iter().cloned().collect();
    if needs_models {
        let used: BTreeSet<Provider> = models.iter().map(|m| m.provider).collect();
        for p in used {
            let settings = config.provider_settings(p);
            endpoints.push(settings.base_url.clone());
            let Some(var) = settings.api_key_env else {
                continue;
            };
            match secret(&var)? {
                Some(k) => {
                    secrets.push(k.clone());
                    keys.insert(p, k);
                }
                None if p.needs_key() => {
                    bail!("{p} needs an API key: set {var} (or {var}_FILE)")
                }
                None => {}
            }
        }
    }
    let llms = Llms::new(&config, &keys)?;

    let mut meta = Meta {
        version: crate::VERSION.to_string(),
        started: jiff::Timestamp::now()
            .strftime("%Y-%m-%d %H:%M UTC")
            .to_string(),
        suites: suites.iter().copied().collect(),
        models: models.iter().map(ModelSpec::label).collect(),
        distractors: distractors.clone(),
        repeat: opts.repeat,
        ..Default::default()
    };

    let classifier = if suites.contains(&Suite::Classifier) {
        if config.approval.base_url.is_none() {
            meta.notes
                .push("classifier suite skipped: the config has no [approval] base_url".into());
            None
        } else {
            let key = match &config.approval.api_key_env {
                Some(var) => secret(var)?,
                None => None,
            };
            if let Some(k) = &key {
                secrets.push(k.clone());
            }
            let c = Classifier::new(&config.approval, key)?;
            meta.classifier = Some(ClassifierMeta {
                label: format!("classifier:{}", config.approval.model),
                threshold: c.threshold(),
                cutoff: c.cutoff(),
            });
            Some(c)
        }
    } else {
        None
    };

    let sink = Sink::create(&opts.output, secrets, &endpoints)?;
    let mut runner = suites::Runner::new(&config, &fixtures, &llms, sink);
    for m in &models {
        for &n in &distractors {
            if suites.contains(&Suite::Agent) {
                runner.agent_suite(m, n, opts.repeat).await?;
            }
            if suites.contains(&Suite::Capture) {
                runner.capture_suite(m, n, opts.repeat).await?;
            }
        }
        if suites.contains(&Suite::Judge) {
            runner.judge_suite(m, opts.repeat).await?;
        }
    }
    if let (Some(c), Some(cm)) = (&classifier, &meta.classifier) {
        runner.classifier_suite(c, &cm.label, opts.repeat).await?;
    }
    let sink = runner.finish();

    let text = sink.redact(&report::render(&meta, &sink.rows));
    std::fs::write(&opts.report, text)
        .with_context(|| format!("writing {}", opts.report.display()))?;
    Ok(Summary {
        rows: sink.rows.len(),
        harmful: report::harmful_count(&sink.rows),
    })
}

/// The tools a profile is offered in the eval: the fixture tools plus `n`
/// distractors, through the production schema check, profile filter and
/// sorting. No tool policies apply (the eval scores what the model
/// proposes; policies are production's safety net).
pub fn offered_tools(
    fixtures: &Fixtures,
    profile: &ProfileConfig,
    n: usize,
) -> Result<Vec<crate::tools::ToolDef>> {
    let discovered = fixtures
        .tools
        .iter()
        .chain(fixtures.distractors(n)?)
        .map(fixtures::ToolFixture::discovered)
        .collect();
    let filter = crate::policy::ToolFilter::new(&profile.tools)?;
    let (tools, skipped) =
        agent::prepare_tools(discovered, &filter, &crate::policy::Policies::new(&[])?);
    if let Some((tool, why)) = skipped.first() {
        bail!("fixture tool {tool} has an unusable schema: {why}");
    }
    Ok(tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_specs_resolve_like_profiles() {
        let cfg = Config::from_toml(
            "[llm.ollama]\nmodel = \"small:4b\"\neffort = \"none\"\n[llm.anthropic]\neffort = \"low\"\n",
        )
        .unwrap();
        let m = ModelSpec::parse("ollama:qwen3.5:9b", &cfg).unwrap();
        assert_eq!(m.provider, Provider::Ollama);
        assert_eq!(m.model, "qwen3.5:9b");
        assert_eq!(m.effort, "none");
        assert_eq!(m.label(), "ollama:qwen3.5:9b");
        assert_eq!(ModelSpec::parse("ollama", &cfg).unwrap().model, "small:4b");
        assert_eq!(
            ModelSpec::parse("anthropic:claude-haiku-5-5", &cfg)
                .unwrap()
                .effort,
            "low"
        );
        assert!(ModelSpec::parse("gemini:x", &cfg).is_err());
        assert!(
            ModelSpec::parse("openai", &cfg).is_err(),
            "no default model"
        );
        assert_eq!(
            ModelSpec::default_for(&cfg).unwrap().label(),
            "anthropic:claude-opus-5-5"
        );
    }

    #[test]
    fn usage_is_read_from_every_provider_shape() {
        let anthropic = json!({"usage": {"input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}});
        let u = Usage::from_audit(&anthropic);
        assert_eq!(u.prompt_tokens, Some(1010));
        assert_eq!(u.cached_tokens, Some(1000));
        let ollama = json!({"usage": {"input_tokens": 1800, "output_tokens": 20, "cached_tokens": null,
            "load_ms": 2.5, "prompt_eval_ms": 900.0, "eval_ms": 300.0, "total_ms": 1210.0}});
        let u = Usage::from_audit(&ollama);
        assert_eq!(u.prompt_tokens, Some(1800));
        assert_eq!(u.prompt_eval_ms, Some(900.0));
        assert_eq!(Usage::from_audit(&json!({})), Usage::default());
    }

    #[test]
    fn endpoints_are_redacted_in_every_spelling() {
        let dir = tempfile::tempdir().unwrap();
        let sink = Sink::create(
            &dir.path().join("r.jsonl"),
            vec!["sk-secret-key".into()],
            &[
                "http://gpu-box.lan:11434/".into(),
                "https://api.example.com/v1".into(),
            ],
        )
        .unwrap();
        assert_eq!(
            sink.redact(
                "error sending request for url (http://gpu-box.lan:11434/api/chat); \
                 gpu-box.lan:11434 down; https://api.example.com/v1/chat/completions; sk-secret-key"
            ),
            "error sending request for url ([endpoint]/api/chat); [endpoint] down; \
             [endpoint]/chat/completions; [redacted]"
        );
    }

    #[test]
    fn suites_parse() {
        assert_eq!(Suite::parse(" Judge").unwrap(), Suite::Judge);
        assert!(Suite::parse("all").is_err());
    }
}
