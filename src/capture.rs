//! The capture guarantee: a profile's `require` says which tool call makes
//! a job a success, so a job never reports success it cannot verify.
//!
//! A job of a profile with `require` succeeds only if a call of the
//! required tool returned a result that is not an error, matches
//! `result_matching` (when given) and none of `result_not_matching`. The
//! model's reply does not count: a small model may say "Added" after a tool
//! that only returned a plan. When the calls do not meet the criteria,
//! Ringmaster can make one call of its own (`fallback`, with the whole
//! transcript), and otherwise fails the job with a notification holding the
//! whole transcript, so nothing the user said is lost.
//!
//! This module holds the compiled criteria ([`Requirement`]), the title and
//! argument templates of the fallback, and what a job keeps about it in
//! its `AgentState` ([`CaptureLog`]). The flow is in `Agent::guarantee`.

use anyhow::{Context, Result, bail};
use globset::GlobMatcher;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{FallbackConfig, RequireConfig};

/// What a job of a profile with `require` keeps about it. `None` in a
/// state (saved before the guarantee existed, or by a profile without
/// `require`) means the job runs exactly as before.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CaptureLog {
    /// Calls of the required tool that met the criteria.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub saved: Vec<SavedCall>,
    /// Calls of the required tool that did not, and why.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected: Vec<RejectedCall>,
    /// Why the job was found unverified (set once; audited once).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unverified: Option<String>,
    /// The fallback in progress (between its calls, or parked for
    /// approval).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<FallbackProgress>,
}

impl CaptureLog {
    /// Whether some call met the criteria.
    pub fn is_saved(&self) -> bool {
        !self.saved.is_empty()
    }

    /// Keep what a call of `tool` counted as.
    pub fn record(&mut self, tool: &str, input: &Value, check: Check, unreachable: bool, by: &str) {
        match check {
            Check::NotRequired => {}
            Check::Saved => self.saved.push(SavedCall {
                tool: tool.to_string(),
                input: input.clone(),
                by: by.to_string(),
            }),
            Check::Rejected(reason) => self.rejected.push(RejectedCall {
                tool: tool.to_string(),
                reason,
                unreachable,
            }),
        }
    }

    /// Whether the last call of the required tool could not reach its
    /// server.
    pub fn unreachable(&self) -> bool {
        self.rejected.last().is_some_and(|r| r.unreachable)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedCall {
    /// `<server>/<tool>`.
    pub tool: String,
    pub input: Value,
    /// `model` or `fallback`.
    pub by: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectedCall {
    /// `<server>/<tool>`.
    pub tool: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unreachable: bool,
}

/// A fallback between its calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FallbackProgress {
    /// 0: the first call; 1: the confirm call after a plan.
    pub step: u8,
    /// Arguments of the call to run next.
    pub input: Value,
    pub title: String,
    /// Parked for Approve/Deny.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub awaiting: bool,
    /// Who approved the fallback when a policy asked for it (`human`); one
    /// approval covers both calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_by: Option<String>,
}

/// Whether a call of the required tool counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    /// Not the required tool.
    NotRequired,
    Saved,
    /// The required tool, but the result does not count.
    Rejected(String),
}

/// A profile's compiled `require`.
#[derive(Debug, Clone)]
pub struct Requirement {
    tool: GlobMatcher,
    pattern: String,
    matching: Vec<(String, GlobMatcher)>,
    not_matching: Vec<(String, GlobMatcher)>,
    pub fallback: Option<Fallback>,
}

/// A compiled `require.fallback`.
#[derive(Debug, Clone)]
pub struct Fallback {
    /// `<server>/<tool>`.
    pub tool: String,
    pub server: String,
    pub name: String,
    pub arguments: Map<String, Value>,
    pub title_max_chars: usize,
    pub confirm: Option<FallbackConfirm>,
}

#[derive(Debug, Clone)]
pub struct FallbackConfirm {
    when: Vec<GlobMatcher>,
    pub arguments: Map<String, Value>,
}

impl FallbackConfirm {
    /// Whether a result is a plan that needs the confirm call.
    pub fn is_plan(&self, text: &str, structured: Option<&Value>) -> bool {
        texts(text, structured).any(|t| self.when.iter().any(|g| g.is_match(&t)))
    }
}

/// Glob over a result: case-insensitive, `*` matches anything (newlines
/// included).
fn result_glob(pattern: &str) -> Result<GlobMatcher> {
    Ok(globset::GlobBuilder::new(pattern)
        .case_insensitive(true)
        .literal_separator(false)
        .backslash_escape(true)
        .build()
        .with_context(|| format!("result glob {pattern:?}"))?
        .compile_matcher())
}

/// The texts a result glob is checked against: the text, and the
/// structured content as JSON (MCP for Cloud keeps `requiresConfirmation`
/// there while its text is translated).
fn texts<'a>(text: &'a str, structured: Option<&'a Value>) -> impl Iterator<Item = String> + 'a {
    std::iter::once(text.to_string()).chain(structured.map(Value::to_string))
}

const PLACEHOLDERS: &[&str] = &["{{title}}", "{{transcript}}"];

impl Requirement {
    /// Compile and check a `require`. `profile_tools` are the profile's
    /// `tools` globs: the fallback must be one of its tools.
    pub fn new(cfg: &RequireConfig, profile_tools: &[String]) -> Result<Self> {
        let tool = crate::policy::tool_glob(&cfg.tool)
            .with_context(|| format!("tool {:?}", cfg.tool))?
            .compile_matcher();
        let compile = |globs: &[String]| {
            globs
                .iter()
                .map(|g| Ok((g.clone(), result_glob(g)?)))
                .collect::<Result<Vec<_>>>()
        };
        let fallback = cfg
            .fallback
            .as_ref()
            .map(|f| compile_fallback(f, &tool, &cfg.tool, profile_tools))
            .transpose()?;
        Ok(Self {
            tool,
            pattern: cfg.tool.clone(),
            matching: compile(&cfg.result_matching)?,
            not_matching: compile(&cfg.result_not_matching)?,
            fallback,
        })
    }

    /// The `tool` glob, as configured.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn applies_to(&self, qualified: &str) -> bool {
        self.tool.is_match(qualified)
    }

    /// Whether a call of `qualified` with this result meets the criteria.
    pub fn check(
        &self,
        qualified: &str,
        text: &str,
        structured: Option<&Value>,
        is_error: bool,
    ) -> Check {
        if !self.applies_to(qualified) {
            return Check::NotRequired;
        }
        if is_error {
            return Check::Rejected("the result was an error".into());
        }
        if let Some((g, _)) = self
            .not_matching
            .iter()
            .find(|(_, m)| texts(text, structured).any(|t| m.is_match(&t)))
        {
            return Check::Rejected(format!("the result matched {g:?}"));
        }
        if !self.matching.is_empty()
            && !self
                .matching
                .iter()
                .any(|(_, m)| texts(text, structured).any(|t| m.is_match(&t)))
        {
            let all: Vec<&str> = self.matching.iter().map(|(g, _)| g.as_str()).collect();
            return Check::Rejected(format!("the result matched none of {all:?}"));
        }
        Check::Saved
    }
}

fn compile_fallback(
    f: &FallbackConfig,
    required: &GlobMatcher,
    pattern: &str,
    profile_tools: &[String],
) -> Result<Fallback> {
    let Some((server, name)) = f.tool.split_once('/') else {
        bail!("fallback.tool {:?} must be <server>/<tool>", f.tool);
    };
    if server.is_empty() || name.is_empty() || f.tool.contains(['*', '?', '[', '{']) {
        bail!(
            "fallback.tool {:?} must name one tool (<server>/<tool>, no globs)",
            f.tool
        );
    }
    if !required.is_match(&f.tool) {
        bail!(
            "fallback.tool {:?} does not match require.tool {pattern:?}",
            f.tool
        );
    }
    let allowed = crate::policy::ToolFilter::new(profile_tools)?.allows(&f.tool);
    if !allowed {
        bail!(
            "fallback.tool {:?} is not in the profile's tools, so it could never run",
            f.tool
        );
    }
    if f.title_max_chars < 8 {
        bail!("fallback.title_max_chars must be at least 8");
    }
    check_placeholders(&Value::Object(f.arguments.clone()), false).context("fallback.arguments")?;
    let confirm = f
        .confirm
        .as_ref()
        .map(|c| {
            if c.when_result_matching.is_empty() {
                bail!("fallback.confirm.when_result_matching must hold at least one glob");
            }
            check_placeholders(&Value::Object(c.arguments.clone()), true)
                .context("fallback.confirm.arguments")?;
            Ok(FallbackConfirm {
                when: c
                    .when_result_matching
                    .iter()
                    .map(|g| result_glob(g))
                    .collect::<Result<_>>()?,
                arguments: c.arguments.clone(),
            })
        })
        .transpose()?;
    Ok(Fallback {
        tool: f.tool.clone(),
        server: server.to_string(),
        name: name.to_string(),
        arguments: f.arguments.clone(),
        title_max_chars: f.title_max_chars,
        confirm,
    })
}

/// Every `{{...}}` in the strings of `v` is a known placeholder
/// (`{{result:/pointer}}` only in the confirm call).
fn check_placeholders(v: &Value, result_allowed: bool) -> Result<()> {
    match v {
        Value::String(s) => {
            let mut rest = s.as_str();
            while let Some(start) = rest.find("{{") {
                let Some(len) = rest[start..].find("}}") else {
                    bail!("unclosed {{{{ in {s:?}");
                };
                let p = &rest[start..start + len + 2];
                let ok =
                    PLACEHOLDERS.contains(&p) || (result_allowed && p.starts_with("{{result:/"));
                if !ok {
                    bail!(
                        "unknown placeholder {p} in {s:?} (use {{{{title}}}}, {{{{transcript}}}}{})",
                        if result_allowed {
                            " or {{result:/pointer}}"
                        } else {
                            ""
                        }
                    );
                }
                rest = &rest[start + len + 2..];
            }
            Ok(())
        }
        Value::Array(items) => items
            .iter()
            .try_for_each(|i| check_placeholders(i, result_allowed)),
        Value::Object(m) => m
            .values()
            .try_for_each(|i| check_placeholders(i, result_allowed)),
        _ => Ok(()),
    }
}

/// What the placeholders stand for.
pub struct Values<'a> {
    pub title: &'a str,
    pub transcript: &'a str,
    /// The first call's result as JSON (its structured content, or its
    /// text parsed as JSON), for `{{result:/pointer}}`.
    pub result: Option<&'a Value>,
}

/// Fill the placeholders of `args`. A string that is exactly one
/// `{{result:/pointer}}` becomes that value whatever its type; elsewhere
/// values are put into the text. A pointer the result does not have is an
/// error.
pub fn fill(args: &Map<String, Value>, v: &Values<'_>) -> Result<Map<String, Value>, String> {
    args.iter()
        .map(|(k, val)| Ok((k.clone(), fill_value(val, v)?)))
        .collect()
}

fn fill_value(val: &Value, v: &Values<'_>) -> Result<Value, String> {
    Ok(match val {
        Value::String(s) => {
            if let Some(ptr) = s
                .strip_prefix("{{result:")
                .and_then(|r| r.strip_suffix("}}"))
                .filter(|p| !p.contains("}}"))
            {
                return lookup(v.result, ptr).cloned();
            }
            let mut out = String::with_capacity(s.len());
            let mut rest = s.as_str();
            while let Some(start) = rest.find("{{") {
                out.push_str(&rest[..start]);
                let len = rest[start..].find("}}").map(|l| l + 2).unwrap_or(2);
                let p = &rest[start..start + len];
                match p {
                    "{{title}}" => out.push_str(v.title),
                    "{{transcript}}" => out.push_str(v.transcript),
                    _ => {
                        let ptr = p
                            .strip_prefix("{{result:")
                            .and_then(|r| r.strip_suffix("}}"))
                            .unwrap_or_default();
                        match lookup(v.result, ptr)? {
                            Value::String(t) => out.push_str(t),
                            other => out.push_str(&other.to_string()),
                        }
                    }
                }
                rest = &rest[start + len..];
            }
            out.push_str(rest);
            Value::String(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|i| fill_value(i, v))
                .collect::<Result<_, _>>()?,
        ),
        Value::Object(m) => Value::Object(fill(m, v)?),
        other => other.clone(),
    })
}

fn lookup<'a>(result: Option<&'a Value>, ptr: &str) -> Result<&'a Value, String> {
    result
        .and_then(|r| r.pointer(ptr))
        .ok_or_else(|| format!("the plan has no {ptr:?} to confirm it with"))
}

/// The first result as JSON, for `{{result:/pointer}}`: the structured
/// content, else the text if it is JSON.
pub fn result_json(text: &str, structured: Option<&Value>) -> Option<Value> {
    structured
        .cloned()
        .or_else(|| serde_json::from_str(text.trim()).ok())
}

/// A card title from the transcript: its first line, then its first
/// sentence, cut at a word boundary to at most `max` characters (with "…"
/// when cut). Never empty.
pub fn title(transcript: &str, max: usize) -> String {
    let line = transcript
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default();
    let sentence = sentence_end(line).map_or(line, |end| &line[..end]).trim();
    let sentence = sentence.trim_end_matches([',', ';', ':', '-', ' ']);
    if sentence.chars().count() <= max {
        return if sentence.is_empty() {
            "Note".into()
        } else {
            sentence.to_string()
        };
    }
    // Room for the ellipsis.
    let limit = max.saturating_sub(1);
    let cut = sentence
        .char_indices()
        .nth(limit)
        .map_or(sentence.len(), |(i, _)| i);
    let head = &sentence[..cut];
    // Back to the last whole word, unless that leaves almost nothing (one
    // very long word).
    let head = match head.rfind(char::is_whitespace) {
        Some(i) if i >= cut / 3 && !sentence[cut..].starts_with(char::is_whitespace) => &head[..i],
        _ => head,
    };
    format!("{}…", head.trim_end_matches([',', ';', ':', '-', ' ', '.']))
}

/// The byte index just after the first sentence's end mark (`.`, `!` or
/// `?` followed by a space or the end), if the line has more after it.
fn sentence_end(line: &str) -> Option<usize> {
    let mut chars = line.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if matches!(c, '.' | '!' | '?') {
            match chars.peek() {
                Some((_, n)) if n.is_whitespace() => return Some(i + c.len_utf8()),
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FallbackConfirmConfig;
    use serde_json::json;

    fn req(tool: &str, matching: &[&str], not: &[&str]) -> RequireConfig {
        RequireConfig {
            tool: tool.into(),
            result_matching: matching.iter().map(|s| s.to_string()).collect(),
            result_not_matching: not.iter().map(|s| s.to_string()).collect(),
            fallback: None,
        }
    }

    const PLAN: &str = "**Create card in Deck**\n\nCreate card in *Inbox* › *Inbox*: **Buy milk**\n\n\
                        Nothing was changed. Confirm to execute.";

    #[test]
    fn plans_errors_and_other_tools_do_not_count() {
        let r = Requirement::new(
            &req(
                "nextcloud/deck_create_card",
                &[],
                &["*Nothing was changed*", "*requiresConfirmation*"],
            ),
            &[],
        )
        .unwrap();
        let q = "nextcloud/deck_create_card";
        assert_eq!(
            r.check(
                q,
                "{\n    \"id\": 7,\n    \"title\": \"Buy milk\"\n}",
                None,
                false
            ),
            Check::Saved
        );
        // Globs reach across lines and ignore case.
        assert_eq!(
            r.check(q, PLAN, None, false),
            Check::Rejected("the result matched \"*Nothing was changed*\"".into())
        );
        assert!(matches!(
            r.check(q, &PLAN.to_uppercase(), None, false),
            Check::Rejected(_)
        ));
        // A translated plan still carries the key in its structured content.
        let structured = json!({"requiresConfirmation": true, "tool": "deck_create_card"});
        assert_eq!(
            r.check(q, "Nada foi alterado.", Some(&structured), false),
            Check::Rejected("the result matched \"*requiresConfirmation*\"".into())
        );
        assert_eq!(
            r.check(q, "{\"id\": 7}", None, true),
            Check::Rejected("the result was an error".into())
        );
        assert_eq!(
            r.check("nextcloud/deck_list_boards", PLAN, None, false),
            Check::NotRequired
        );
        // `*` in the tool glob does not cross the `/`.
        let r = Requirement::new(&req("*_create_card", &[], &[]), &[]).unwrap();
        assert_eq!(r.check(q, "ok", None, false), Check::NotRequired);
    }

    #[test]
    fn result_matching_must_match() {
        let r = Requirement::new(&req("deck/*", &["*\"id\"*"], &[]), &[]).unwrap();
        assert_eq!(
            r.check("deck/create", "{\"id\": 1}", None, false),
            Check::Saved
        );
        assert!(matches!(
            r.check("deck/create", "OK", None, false),
            Check::Rejected(_)
        ));
    }

    fn fallback(tool: &str, args: Value) -> FallbackConfig {
        FallbackConfig {
            tool: tool.into(),
            arguments: args.as_object().cloned().unwrap(),
            title_max_chars: 80,
            confirm: None,
        }
    }

    #[test]
    fn fallback_is_checked_against_the_requirement_and_the_profile() {
        let tools = vec!["nextcloud/deck_*".to_string()];
        let mut cfg = req("nextcloud/deck_create_card", &[], &[]);
        let ok = json!({"stackId": 49, "title": "{{title}}", "description": "{{transcript}}"});
        cfg.fallback = Some(fallback("nextcloud/deck_create_card", ok.clone()));
        assert!(Requirement::new(&cfg, &tools).is_ok());
        for (tool, args, why) in [
            ("nextcloud/deck_*", ok.clone(), "no globs"),
            ("deck_create_card", ok.clone(), "<server>/<tool>"),
            ("nextcloud/deck_delete_card", ok.clone(), "does not match"),
            (
                "nextcloud/deck_create_card",
                json!({"title": "{{name}}"}),
                "unknown placeholder",
            ),
            (
                "nextcloud/deck_create_card",
                json!({"plan_state": "{{result:/plan_state}}"}),
                "unknown placeholder",
            ),
        ] {
            cfg.fallback = Some(fallback(tool, args));
            let e = format!("{:#}", Requirement::new(&cfg, &tools).unwrap_err());
            assert!(e.contains(why), "{tool}: {e}");
        }
        cfg.fallback = Some(fallback("nextcloud/deck_create_card", ok));
        let e = format!("{:#}", Requirement::new(&cfg, &[]).unwrap_err());
        assert!(e.contains("not in the profile's tools"), "{e}");
        // The confirm call may use the plan.
        let mut f = cfg.fallback.clone().unwrap();
        f.confirm = Some(FallbackConfirmConfig {
            when_result_matching: vec!["*requiresConfirmation*".into()],
            arguments: json!({"confirm": true, "plan_state": "{{result:/plan_state}}"})
                .as_object()
                .cloned()
                .unwrap(),
        });
        cfg.fallback = Some(f);
        let r = Requirement::new(&cfg, &tools).unwrap();
        let c = r.fallback.unwrap().confirm.unwrap();
        assert!(c.is_plan("x", Some(&json!({"requiresConfirmation": true}))));
        assert!(!c.is_plan("{\"id\": 1}", None));
    }

    #[test]
    fn placeholders_are_filled_and_keep_types() {
        let args = json!({
            "stackId": 49,
            "title": "{{title}}",
            "description": "Said: {{transcript}}",
            "confirm": true,
            "plan_state": "{{result:/plan_state}}",
            "note": "plan {{result:/plan_state}} for {{result:/n}}",
            "n": "{{result:/n}}",
            "nested": ["{{title}}"],
        });
        let result = json!({"plan_state": "ab12", "n": 3});
        let v = Values {
            title: "Buy milk",
            transcript: "buy milk. and eggs",
            result: Some(&result),
        };
        let out = fill(args.as_object().unwrap(), &v).unwrap();
        assert_eq!(
            Value::Object(out),
            json!({
                "stackId": 49,
                "title": "Buy milk",
                "description": "Said: buy milk. and eggs",
                "confirm": true,
                "plan_state": "ab12",
                "note": "plan ab12 for 3",
                "n": 3,
                "nested": ["Buy milk"],
            })
        );
        let missing = fill(
            json!({"plan_state": "{{result:/plan_state}}"})
                .as_object()
                .unwrap(),
            &Values { result: None, ..v },
        );
        assert!(missing.unwrap_err().contains("/plan_state"));
        assert_eq!(result_json("{\"a\": 1}", None), Some(json!({"a": 1})));
        assert_eq!(result_json("not json", None), None);
        assert_eq!(
            result_json("{\"a\": 1}", Some(&json!({"b": 2}))),
            Some(json!({"b": 2}))
        );
    }

    #[test]
    fn titles_come_from_the_first_sentence_and_stop_at_a_word() {
        assert_eq!(title("  buy milk  ", 80), "buy milk");
        assert_eq!(
            title("Call the plumber. The tap drips at night.", 80),
            "Call the plumber."
        );
        assert_eq!(title("\n\nfirst line\nsecond line", 80), "first line");
        assert_eq!(
            title("version 2.5 of the plan", 80),
            "version 2.5 of the plan"
        );
        let long = "so I was thinking we should probably sort out the gutters before the winter because last year";
        let t = title(long, 40);
        assert!(t.chars().count() <= 40, "{t}");
        assert_eq!(t, "so I was thinking we should probably…");
        // One very long word is cut where it has to be.
        let t = title(&"a".repeat(100), 10);
        assert_eq!(t, format!("{}…", "a".repeat(9)));
        assert_eq!(title("   ", 80), "Note");
        assert_eq!(title("ok, ", 80), "ok");
        // Characters, not bytes.
        let t = title(&"é ".repeat(60), 20);
        assert!(t.chars().count() <= 20, "{t}");
    }
}
