//! Eval fixtures: tool listings, profiles, scripted tool results and the
//! cases of each suite. Everything is synthetic (a made-up household); see
//! `eval/cases/` and `docs/model-evaluation.md`.
//!
//! Expectations are written exactly like `[[tool_policies]]` rules (a glob
//! over `<server>/<tool>` plus `when`/`when_any`/`unless` JSON-pointer
//! conditions) and matched by the same code ([`Policies::first_match`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::config::{ProfileConfig, ToolPolicyRule};
use crate::policy::{Policies, Policy};
use crate::tools::DiscoveredTool;

/// One rule: a tool glob and optional argument conditions, as in a
/// `[[tool_policies]]` entry (without `policy`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matcher {
    #[serde(rename = "match")]
    pub pattern: String,
    #[serde(default)]
    pub when: BTreeMap<String, String>,
    #[serde(default)]
    pub when_any: BTreeMap<String, String>,
    #[serde(default)]
    pub unless: BTreeMap<String, String>,
}

impl Matcher {
    fn rule(&self) -> ToolPolicyRule {
        let mut r = ToolPolicyRule::new(&self.pattern, Policy::Allow);
        r.when = self.when.clone();
        r.when_any = self.when_any.clone();
        r.unless = self.unless.clone();
        r
    }
}

/// A list of matchers; a call matches the set when any of them matches.
#[derive(Debug, Clone)]
pub struct MatchSet {
    policies: Policies,
    patterns: Vec<String>,
}

impl MatchSet {
    pub fn new(matchers: &[Matcher]) -> Result<Self> {
        let rules: Vec<ToolPolicyRule> = matchers.iter().map(Matcher::rule).collect();
        Ok(Self {
            policies: Policies::new(&rules)?,
            patterns: matchers.iter().map(|m| m.pattern.clone()).collect(),
        })
    }

    pub fn empty() -> Self {
        Self::new(&[]).expect("an empty set builds")
    }

    /// Whether a call of `<server>/<tool>` with `args` matches.
    pub fn matches(&self, qualified: &str, args: &Value) -> bool {
        self.policies.first_match(qualified, args).is_some()
    }

    /// Which matcher matches, if any.
    pub fn position(&self, qualified: &str, args: &Value) -> Option<usize> {
        self.policies.first_match(qualified, args)
    }

    /// Patterns that match none of `tools` by name.
    fn dangling(&self, tools: &[String]) -> Result<Vec<&str>> {
        let mut out = Vec::new();
        for p in &self.patterns {
            let by_name = Policies::new(&[ToolPolicyRule::new(p, Policy::Allow)])?;
            if !tools
                .iter()
                .any(|t| by_name.first_match(t, &Value::Null).is_some())
            {
                out.push(p.as_str());
            }
        }
        Ok(out)
    }
}

/// A tool as an MCP server would list it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolFixture {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Map<String, Value>,
}

impl ToolFixture {
    pub fn qualified(&self) -> String {
        format!("{}/{}", self.server, self.name)
    }

    pub fn discovered(&self) -> DiscoveredTool {
        DiscoveredTool {
            server: self.server.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }
}

/// A scripted tool result: the first entry matching the call answers it.
/// `{{/pointer}}` in `text` is replaced by that argument's value.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultFixture {
    #[serde(rename = "match")]
    pub pattern: String,
    #[serde(default)]
    pub when: BTreeMap<String, String>,
    #[serde(default)]
    pub when_any: BTreeMap<String, String>,
    #[serde(default)]
    pub unless: BTreeMap<String, String>,
    pub text: String,
    /// Sent back as a failed tool call.
    #[serde(default)]
    pub error: bool,
}

impl ResultFixture {
    fn matcher(&self) -> Matcher {
        Matcher {
            pattern: self.pattern.clone(),
            when: self.when.clone(),
            when_any: self.when_any.clone(),
            unless: self.unless.clone(),
        }
    }
}

/// Scripted results: a case's own entries first, then the defaults.
#[derive(Debug, Clone)]
pub struct Script {
    entries: Vec<ResultFixture>,
    set: MatchSet,
}

/// Answer for a call no scripted result matches.
pub const DEFAULT_RESULT: &str = "OK";

impl Script {
    pub fn new(case: &[ResultFixture], defaults: &[ResultFixture]) -> Result<Self> {
        let entries: Vec<ResultFixture> = case.iter().chain(defaults).cloned().collect();
        let matchers: Vec<Matcher> = entries.iter().map(ResultFixture::matcher).collect();
        Ok(Self {
            set: MatchSet::new(&matchers)?,
            entries,
        })
    }

    /// The result text for a call, and whether it is an error.
    pub fn answer(&self, qualified: &str, args: &Value) -> (String, bool) {
        match self.set.position(qualified, args) {
            Some(i) => {
                let e = &self.entries[i];
                (fill_template(&e.text, args), e.error)
            }
            None => (DEFAULT_RESULT.to_string(), false),
        }
    }
}

/// Replace `{{/pointer}}` with the argument's value: strings JSON-escaped
/// without quotes (so they fit inside a JSON string), anything else as JSON
/// text, `null` when missing.
pub fn fill_template(text: &str, args: &Value) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{{/") {
        let Some(len) = rest[start..].find("}}") else {
            break;
        };
        out.push_str(&rest[..start]);
        let pointer = &rest[start + 2..start + len];
        let value = match args.pointer(pointer) {
            Some(Value::String(s)) => {
                let quoted = serde_json::to_string(s).unwrap_or_default();
                quoted[1..quoted.len() - 1].to_string()
            }
            Some(v) => v.to_string(),
            None => "null".into(),
        };
        out.push_str(&value);
        rest = &rest[start + len + 2..];
    }
    out.push_str(rest);
    out
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilesFile {
    profiles: BTreeMap<String, ProfileConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultsFile {
    #[serde(default)]
    result: Vec<ResultFixture>,
}

/// `agent.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentSuite {
    /// Model calls per case at most (lookups answered in between).
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    /// Read-only calls that are answered with scripted results instead of
    /// being scored, so the model can look things up before acting.
    #[serde(default)]
    pub lookups: Vec<Matcher>,
    /// Harmful in every case unless the case expects it.
    #[serde(default)]
    pub forbid: Vec<Matcher>,
    /// Named lists of matchers that cases can forbid by name.
    #[serde(default)]
    pub groups: BTreeMap<String, Vec<Matcher>>,
    #[serde(default, rename = "case")]
    pub cases: Vec<AgentCase>,
}

fn default_max_turns() -> u32 {
    5
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentCase {
    pub id: String,
    pub category: String,
    /// The `X-Index-Trigger` value; selects the profile.
    pub gesture: String,
    pub transcript: String,
    /// Transcript source shown to the model (default `primary`).
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub expect: Vec<Matcher>,
    /// Harmful, even if an `expect` matcher also matches.
    #[serde(default)]
    pub forbid: Vec<Matcher>,
    /// Groups from the suite's `[groups]` that are harmful in this case.
    #[serde(default)]
    pub forbid_groups: Vec<String>,
    /// Doing nothing is a pass.
    #[serde(default)]
    pub allow_no_call: bool,
    #[serde(default)]
    pub results: Vec<ResultFixture>,
    #[serde(default)]
    pub note: Option<String>,
}

/// `capture.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSuite {
    pub gesture: String,
    /// Calls that create a card.
    pub create: Matcher,
    /// A created card that is in the right list.
    pub inbox: Matcher,
    /// Where the card title is in the create call's arguments.
    #[serde(default = "default_title_pointer")]
    pub title_pointer: String,
    /// Reply phrases that claim a card was added (case-insensitive).
    pub success_claims: Vec<String>,
    #[serde(default = "default_overlap")]
    pub min_title_overlap: f64,
    #[serde(default, rename = "case")]
    pub cases: Vec<CaptureCase>,
}

fn default_title_pointer() -> String {
    "/title".into()
}

fn default_overlap() -> f64 {
    0.5
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureExpect {
    /// Exactly one card in the inbox.
    Card,
    /// No card (junk, cancelled, or the Deck fails).
    None,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureCase {
    pub id: String,
    pub category: String,
    pub transcript: String,
    pub expect: CaptureExpect,
    #[serde(default)]
    pub results: Vec<ResultFixture>,
    #[serde(default)]
    pub note: Option<String>,
}

/// `judge.toml`: proposed calls in pairs, one that matches the request and
/// one that does not.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeSuite {
    #[serde(default, rename = "pair")]
    pub pairs: Vec<JudgePair>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgePair {
    pub id: String,
    /// `clear`, `borderline` (a misheard request mapped to a related but
    /// different device) or `excess` (more than was asked).
    pub category: String,
    pub transcript: String,
    #[serde(default)]
    pub assistant_text: Option<String>,
    pub safe: ProposedFixture,
    #[serde(rename = "unsafe")]
    pub unsafe_call: ProposedFixture,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposedFixture {
    /// `<server>/<tool>`.
    pub tool: String,
    pub arguments: Value,
}

/// Every fixture of a cases directory.
#[derive(Debug, Clone)]
pub struct Fixtures {
    pub dir: PathBuf,
    pub tools: Vec<ToolFixture>,
    pub distractors: Vec<ToolFixture>,
    pub profiles: BTreeMap<String, ProfileConfig>,
    pub results: Vec<ResultFixture>,
    pub agent: AgentSuite,
    pub capture: CaptureSuite,
    pub judge: JudgeSuite,
}

fn read(dir: &Path, name: &str) -> Result<String> {
    let path = dir.join(name);
    std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))
}

fn toml_file<T: serde::de::DeserializeOwned>(dir: &Path, name: &str) -> Result<T> {
    toml::from_str(&read(dir, name)?).with_context(|| format!("parsing {name}"))
}

fn json_tools(dir: &Path, name: &str) -> Result<Vec<ToolFixture>> {
    serde_json::from_str(&read(dir, name)?).with_context(|| format!("parsing {name}"))
}

impl Fixtures {
    /// Load and check every fixture in `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let f = Self {
            dir: dir.to_path_buf(),
            tools: json_tools(dir, "tools.json")?,
            distractors: json_tools(dir, "distractors.json")?,
            profiles: toml_file::<ProfilesFile>(dir, "profiles.toml")?.profiles,
            results: toml_file::<ResultsFile>(dir, "results.toml")?.result,
            agent: toml_file(dir, "agent.toml")?,
            capture: toml_file(dir, "capture.toml")?,
            judge: toml_file(dir, "judge.toml")?,
        };
        f.check()?;
        Ok(f)
    }

    /// The profile a gesture selects.
    pub fn profile_for(&self, gesture: &str) -> Option<(&str, &ProfileConfig)> {
        self.profiles
            .iter()
            .find(|(_, p)| p.triggers.iter().any(|t| t == gesture))
            .map(|(n, p)| (n.as_str(), p))
    }

    pub fn tool(&self, qualified: &str) -> Option<&ToolFixture> {
        self.tools.iter().find(|t| t.qualified() == qualified)
    }

    /// The first `n` distractors (always the same ones, in file order).
    pub fn distractors(&self, n: usize) -> Result<&[ToolFixture]> {
        if n > self.distractors.len() {
            bail!(
                "--distractors {n}: there are only {} distractor tools in {}",
                self.distractors.len(),
                self.dir.join("distractors.json").display()
            );
        }
        Ok(&self.distractors[..n])
    }

    /// Matchers that are harmful in an agent case.
    pub fn agent_forbid(&self, case: &AgentCase) -> Result<Vec<Matcher>> {
        let mut out = case.forbid.clone();
        for g in &case.forbid_groups {
            match self.agent.groups.get(g) {
                Some(ms) => out.extend(ms.iter().cloned()),
                None => bail!("agent case {}: unknown group {g:?}", case.id),
            }
        }
        Ok(out)
    }

    /// Consistency checks: unique ids, known gestures, known tools,
    /// schemas the APIs accept, matchers that parse.
    pub fn check(&self) -> Result<()> {
        let names: Vec<String> = self.tools.iter().map(ToolFixture::qualified).collect();
        let mut seen = BTreeSet::new();
        for t in self.tools.iter().chain(&self.distractors) {
            if !seen.insert(t.qualified()) {
                bail!("tool {} is listed twice", t.qualified());
            }
            crate::tools::check_schema(&t.input_schema)
                .map_err(|e| anyhow::anyhow!("tool {}: {e}", t.qualified()))?;
        }
        let known = |what: &str, ms: &[Matcher]| -> Result<()> {
            let set = MatchSet::new(ms).with_context(|| what.to_string())?;
            let dangling = set.dangling(&names)?;
            if !dangling.is_empty() {
                bail!("{what}: no fixture tool matches {dangling:?}");
            }
            Ok(())
        };
        let results = |what: &str, rs: &[ResultFixture]| -> Result<()> {
            let ms: Vec<Matcher> = rs.iter().map(ResultFixture::matcher).collect();
            known(what, &ms)
        };
        results("results.toml", &self.results)?;
        known("agent.toml lookups", &self.agent.lookups)?;
        known("agent.toml forbid", &self.agent.forbid)?;
        for (g, ms) in &self.agent.groups {
            known(&format!("agent.toml group {g}"), ms)?;
        }
        let mut ids = BTreeSet::new();
        for c in &self.agent.cases {
            if !ids.insert(format!("agent/{}", c.id)) {
                bail!("agent case id {} is used twice", c.id);
            }
            if self.profile_for(&c.gesture).is_none() {
                bail!(
                    "agent case {}: no profile has gesture {:?}",
                    c.id,
                    c.gesture
                );
            }
            if c.transcript.is_empty() {
                bail!("agent case {}: empty transcript", c.id);
            }
            if c.expect.is_empty() && !c.allow_no_call {
                bail!(
                    "agent case {}: nothing expected and no_call not allowed",
                    c.id
                );
            }
            known(&format!("agent case {} expect", c.id), &c.expect)?;
            known(
                &format!("agent case {} forbid", c.id),
                &self.agent_forbid(c)?,
            )?;
            results(&format!("agent case {} results", c.id), &c.results)?;
        }
        let cap = &self.capture;
        if self.profile_for(&cap.gesture).is_none() {
            bail!("capture.toml: no profile has gesture {:?}", cap.gesture);
        }
        known("capture.toml create", std::slice::from_ref(&cap.create))?;
        known("capture.toml inbox", std::slice::from_ref(&cap.inbox))?;
        if cap.success_claims.iter().any(|c| c.trim().is_empty()) {
            bail!("capture.toml: empty success claim");
        }
        for c in &cap.cases {
            if !ids.insert(format!("capture/{}", c.id)) {
                bail!("capture case id {} is used twice", c.id);
            }
            results(&format!("capture case {} results", c.id), &c.results)?;
        }
        for p in &self.judge.pairs {
            if !ids.insert(format!("judge/{}", p.id)) {
                bail!("judge pair id {} is used twice", p.id);
            }
            for (side, call) in [("safe", &p.safe), ("unsafe", &p.unsafe_call)] {
                if self.tool(&call.tool).is_none() {
                    bail!("judge pair {} {side}: unknown tool {}", p.id, call.tool);
                }
                if !call.arguments.is_object() {
                    bail!("judge pair {} {side}: arguments must be a table", p.id);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn m(pattern: &str, when: &[(&str, &str)]) -> Matcher {
        Matcher {
            pattern: pattern.into(),
            when: when
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            when_any: BTreeMap::new(),
            unless: BTreeMap::new(),
        }
    }

    #[test]
    fn matchers_work_like_policy_rules() {
        let set = MatchSet::new(&[
            m(
                "homeassistant/call_action",
                &[("/action", "alarm_control_panel.alarm_arm_away")],
            ),
            m("web/*", &[]),
        ])
        .unwrap();
        // Case-insensitive globs; lists match if any element does.
        assert!(set.matches(
            "homeassistant/call_action",
            &json!({"action": "Alarm_Control_Panel.Alarm_Arm_Away", "target": {"entity_id": ["x"]}})
        ));
        assert!(!set.matches(
            "homeassistant/call_action",
            &json!({"action": "alarm_control_panel.alarm_disarm"})
        ));
        // A missing pointer never matches; `*` does not cross `/`.
        assert!(!set.matches("homeassistant/call_action", &json!({})));
        assert!(set.matches("web/search", &json!({})));
        assert!(!set.matches("web/x/y", &json!({})));
        assert_eq!(set.position("web/search", &json!({})), Some(1));
    }

    #[test]
    fn dangling_patterns_are_found() {
        let set = MatchSet::new(&[
            m("web/search", &[]),
            m("nextcloud/deck_*", &[]),
            m("x/y", &[]),
        ])
        .unwrap();
        let tools = vec![
            "web/search".to_string(),
            "nextcloud/deck_list_boards".into(),
        ];
        assert_eq!(set.dangling(&tools).unwrap(), vec!["x/y"]);
    }

    #[test]
    fn templates_fill_arguments_safely() {
        let args = json!({"title": "Say \"hi\"", "stack_id": 11});
        assert_eq!(
            fill_template(
                r#"{"title":"{{/title}}","stack":{{/stack_id}},"x":{{/missing}}}"#,
                &args
            ),
            r#"{"title":"Say \"hi\"","stack":11,"x":null}"#
        );
        assert_eq!(fill_template("no {{ templates", &args), "no {{ templates");
    }

    #[test]
    fn scripted_results_prefer_the_case_then_defaults() {
        let r = |pattern: &str, text: &str, error: bool| ResultFixture {
            pattern: pattern.into(),
            when: BTreeMap::new(),
            when_any: BTreeMap::new(),
            unless: BTreeMap::new(),
            text: text.into(),
            error,
        };
        let s = Script::new(
            &[r("nextcloud/deck_create_card", "Deck is down", true)],
            &[
                r("nextcloud/deck_create_card", "created {{/title}}", false),
                r("web/*", "results", false),
            ],
        )
        .unwrap();
        assert_eq!(
            s.answer("nextcloud/deck_create_card", &json!({"title": "x"})),
            ("Deck is down".into(), true)
        );
        assert_eq!(
            s.answer("web/search", &json!({})),
            ("results".into(), false)
        );
        assert_eq!(
            s.answer("other/tool", &json!({})),
            (DEFAULT_RESULT.into(), false)
        );
    }
}
