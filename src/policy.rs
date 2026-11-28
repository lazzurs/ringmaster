//! Tool safety policies and per-profile tool allow-lists.
//!
//! Tools are named `<server>/<tool>` for matching. Policy rules are checked in
//! order and the first match wins; a tool no rule matches needs confirmation.
//! A rule may also look at the call's arguments (`when`, `when_any`,
//! `unless`: JSON pointer -> glob), so one tool can get different policies
//! for different calls.
//!
//! `confirm` and `ask` both pause the job for Approve/Deny. They differ only
//! when an approval classifier is configured (`[approval]`): it may approve a
//! `confirm` call on its own, but never an `ask` call, and it never sees a
//! `deny` or `code` call at all. `code` runs a call only when the job carries
//! a valid spoken authorisation code (`src/authcode.rs`).

use anyhow::{Context, Result};
use globset::{Glob, GlobMatcher, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{OnMissing, ToolPolicyRule};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    /// Run without asking.
    Allow,
    /// Pause the job and ask on the phone first (or, in `classifier`
    /// approval mode, let the classifier approve it).
    Confirm,
    /// Always ask on the phone, whatever the approval mode: the classifier
    /// is never asked about these.
    Ask,
    /// Never run. Unconditional `deny` tools are not even offered to the
    /// model; a conditional one is refused when called.
    Deny,
    /// Run only with a valid spoken authorisation code in the same request
    /// (no Approve/Deny, no classifier); otherwise `on_missing`.
    Code,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Allow => "allow",
            Policy::Confirm => "confirm",
            Policy::Ask => "ask",
            Policy::Deny => "deny",
            Policy::Code => "code",
        }
    }
}

/// What a `code` rule needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeRequirement {
    /// The `[auth_codes.codes]` entry.
    pub code: String,
    pub on_missing: OnMissing,
}

/// The rule that decided a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matched<'a> {
    pub policy: Policy,
    /// Set for `code` rules.
    pub code: Option<&'a CodeRequirement>,
    /// The rule looked at the arguments.
    pub conditional: bool,
}

#[derive(Debug, Clone)]
struct Condition {
    pointer: String,
    glob: GlobMatcher,
}

impl Condition {
    fn new(pointer: &str, glob: &str) -> Result<Self> {
        // Home Assistant lowercases actions and entity ids, so matching
        // ignores case: `Alarm_Control_Panel.Alarm_Disarm` must not slip
        // past a rule for `alarm_control_panel.alarm_disarm`.
        let glob = globset::GlobBuilder::new(glob)
            .case_insensitive(true)
            .build()
            .with_context(|| format!("condition {pointer:?}"))?
            .compile_matcher();
        Ok(Self {
            pointer: pointer.to_string(),
            glob,
        })
    }

    /// The value at the pointer, as the texts of its elements (`None` for
    /// an element that is an object or a list, which never matches).
    fn elements(&self, args: &Value) -> Option<Vec<Option<String>>> {
        let v = args.pointer(&self.pointer)?;
        Some(match v {
            Value::Array(items) => items.iter().map(text).collect(),
            other => vec![text(other)],
        })
    }

    /// Some element matches (`when`, `when_any`).
    fn any(&self, args: &Value) -> bool {
        self.elements(args).is_some_and(|els| {
            els.iter()
                .any(|e| e.as_deref().is_some_and(|t| self.glob.is_match(t)))
        })
    }

    /// The value is there and every element matches (`unless`).
    fn all(&self, args: &Value) -> bool {
        self.elements(args).is_some_and(|els| {
            !els.is_empty()
                && els
                    .iter()
                    .all(|e| e.as_deref().is_some_and(|t| self.glob.is_match(t)))
        })
    }
}

/// Strings as they are, numbers and booleans as their JSON text.
fn text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null => Some("null".into()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

#[derive(Debug, Clone)]
struct Rule {
    tool: GlobMatcher,
    when: Vec<Condition>,
    when_any: Vec<Condition>,
    unless: Vec<Condition>,
    policy: Policy,
    code: Option<CodeRequirement>,
}

impl Rule {
    fn conditional(&self) -> bool {
        !(self.when.is_empty() && self.when_any.is_empty() && self.unless.is_empty())
    }

    fn matches(&self, qualified: &str, args: &Value) -> bool {
        self.tool.is_match(qualified)
            && self.when.iter().all(|c| c.any(args))
            && (self.when_any.is_empty() || self.when_any.iter().any(|c| c.any(args)))
            && (self.unless.is_empty() || !self.unless.iter().all(|c| c.all(args)))
    }
}

#[derive(Debug, Clone)]
pub struct Policies {
    rules: Vec<Rule>,
}

impl Policies {
    pub fn new(rules: &[ToolPolicyRule]) -> Result<Self> {
        Self::with_codes(rules, &[])
    }

    /// `codes`: the configured code names, for `code` rules that leave the
    /// name out (allowed when there is exactly one).
    pub fn with_codes(rules: &[ToolPolicyRule], codes: &[String]) -> Result<Self> {
        let conditions = |m: &std::collections::BTreeMap<String, String>| {
            m.iter()
                .map(|(p, g)| Condition::new(p, g))
                .collect::<Result<Vec<_>>>()
        };
        let rules = rules
            .iter()
            .map(|r| {
                let code = (r.policy == Policy::Code).then(|| CodeRequirement {
                    code: r
                        .code
                        .clone()
                        .or_else(|| (codes.len() == 1).then(|| codes[0].clone()))
                        .unwrap_or_default(),
                    on_missing: r.on_missing.unwrap_or_default(),
                });
                Ok(Rule {
                    tool: glob(&r.pattern)?.compile_matcher(),
                    when: conditions(&r.when)?,
                    when_any: conditions(&r.when_any)?,
                    unless: conditions(&r.unless)?,
                    policy: r.policy,
                    code,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { rules })
    }

    /// The rule for a call of `<server>/<tool>` with `args`; calls no rule
    /// matches need confirmation.
    pub fn matched(&self, qualified: &str, args: &Value) -> Matched<'_> {
        self.first_match(qualified, args)
            .map(|i| &self.rules[i])
            .map(|r| Matched {
                policy: r.policy,
                code: r.code.as_ref(),
                conditional: r.conditional(),
            })
            .unwrap_or(Matched {
                policy: Policy::Confirm,
                code: None,
                conditional: false,
            })
    }

    /// The index of the first rule matching a call of `<server>/<tool>`
    /// with `args`, if any. The model evaluation uses this to match calls
    /// against expectations written like policy rules.
    pub fn first_match(&self, qualified: &str, args: &Value) -> Option<usize> {
        self.rules.iter().position(|r| r.matches(qualified, args))
    }

    /// Policy for a call of `<server>/<tool>` with `args`.
    pub fn policy_for(&self, qualified: &str, args: &Value) -> Policy {
        self.matched(qualified, args).policy
    }

    /// Whether every call of the tool is denied, whatever its arguments:
    /// the first rule for it that looks at the tool name alone is `deny`,
    /// and no rule before it looks at the arguments. Only such tools are
    /// hidden from the model; a conditional `deny` is enforced per call.
    pub fn always_denies(&self, qualified: &str) -> bool {
        self.rules
            .iter()
            .find(|r| r.tool.is_match(qualified))
            .is_some_and(|r| !r.conditional() && r.policy == Policy::Deny)
    }
}

/// Which tools a profile may see.
#[derive(Debug, Clone)]
pub struct ToolFilter {
    set: GlobSet,
    /// Server names the profile can reach, `None` meaning any.
    servers: Option<Vec<String>>,
}

impl ToolFilter {
    pub fn new(globs: &[String]) -> Result<Self> {
        let mut b = GlobSetBuilder::new();
        let mut servers = Some(Vec::new());
        for g in globs {
            b.add(glob(g)?);
            let server = g.split('/').next().unwrap_or_default();
            let literal = !server.is_empty() && !server.contains(['*', '?', '[', '{']);
            match (&mut servers, literal) {
                (Some(list), true) => list.push(server.to_string()),
                _ => servers = None,
            }
        }
        Ok(Self {
            set: b.build()?,
            servers,
        })
    }

    pub fn allows(&self, qualified: &str) -> bool {
        self.set.is_match(qualified)
    }

    /// Whether tools from `server` could pass this filter at all. Used to
    /// decide which MCP servers a job really needs.
    pub fn may_use_server(&self, server: &str) -> bool {
        match &self.servers {
            None => !self.set.is_empty(),
            Some(list) => list.iter().any(|s| s == server),
        }
    }
}

fn glob(pattern: &str) -> Result<Glob> {
    // `*` must not cross the `/` between server and tool.
    Ok(globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    fn rule(p: &str, policy: Policy) -> ToolPolicyRule {
        ToolPolicyRule::new(p, policy)
    }

    fn when(p: &str, policy: Policy, conds: &[(&str, &str)]) -> ToolPolicyRule {
        let mut r = rule(p, policy);
        r.when = conds
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        r
    }

    fn none() -> Value {
        json!({})
    }

    #[test]
    fn first_match_gives_the_rule_index() {
        let p = Policies::new(&[
            when("ha/call_action", Policy::Deny, &[("/action", "lock.*")]),
            rule("ha/*", Policy::Allow),
        ])
        .unwrap();
        assert_eq!(
            p.first_match("ha/call_action", &json!({"action": "Lock.Unlock"})),
            Some(0)
        );
        assert_eq!(p.first_match("ha/call_action", &json!({})), Some(1));
        assert_eq!(p.first_match("web/search", &json!({})), None);
        // `matched` still defaults to confirm when nothing matches.
        assert_eq!(p.policy_for("web/search", &json!({})), Policy::Confirm);
    }

    #[test]
    fn first_match_wins_and_default_is_confirm() {
        let p = Policies::new(&[
            rule("homey/*lock*", Policy::Confirm),
            rule("*/*delete*", Policy::Deny),
            rule("homey/*", Policy::Allow),
            rule("nextcloud/deck_get_*", Policy::Allow),
        ])
        .unwrap();
        assert_eq!(p.policy_for("homey/unlock_door", &none()), Policy::Confirm);
        assert_eq!(p.policy_for("homey/set_light", &none()), Policy::Allow);
        assert_eq!(p.policy_for("homey/delete_flow", &none()), Policy::Deny);
        assert_eq!(
            p.policy_for("nextcloud/deck_delete_card", &none()),
            Policy::Deny
        );
        assert_eq!(
            p.policy_for("nextcloud/deck_get_boards", &none()),
            Policy::Allow
        );
        assert_eq!(
            p.policy_for("nextcloud/deck_create_card", &none()),
            Policy::Confirm
        );
    }

    #[test]
    fn ask_is_its_own_policy() {
        let p = Policies::new(&[
            rule("homey/*alarm*", Policy::Ask),
            rule("homey/*", Policy::Confirm),
        ])
        .unwrap();
        assert_eq!(p.policy_for("homey/disarm_alarm", &none()), Policy::Ask);
        assert_eq!(p.policy_for("homey/set_light", &none()), Policy::Confirm);
        let parsed: Policy = serde_json::from_value(serde_json::json!("ask")).unwrap();
        assert_eq!(parsed, Policy::Ask);
        assert_eq!(Policy::Ask.as_str(), "ask");
    }

    #[test]
    fn star_does_not_cross_server_separator() {
        let p = Policies::new(&[rule("homey*", Policy::Allow)]).unwrap();
        assert_eq!(p.policy_for("homey/set_light", &none()), Policy::Confirm);
    }

    #[test]
    fn tool_filter_servers() {
        let f = ToolFilter::new(&["nextcloud/deck_create_card".into()]).unwrap();
        assert!(f.allows("nextcloud/deck_create_card"));
        assert!(!f.allows("nextcloud/deck_delete_card"));
        assert!(f.may_use_server("nextcloud"));
        assert!(!f.may_use_server("homey"));

        let all = ToolFilter::new(&["*/*".into()]).unwrap();
        assert!(all.allows("homey/anything"));
        assert!(all.may_use_server("homey"));

        let none = ToolFilter::new(&[]).unwrap();
        assert!(!none.allows("homey/x"));
        assert!(!none.may_use_server("homey"));
    }

    const CALL: &str = "homeassistant/homeassistant__call_action";
    const UNIFI: &str = "alarm_control_panel.loft_dream_machine_pro_alarm_manager";
    const HOMESECURE: &str = "{alarm_control_panel.64_104058,alarm_control_panel.64_104058_perimeter_only,\
alarm_control_panel.64_104058_motions_only,alarm_control_panel.64_104058_shed_only}";

    #[test]
    fn conditions_use_json_pointers_into_nested_arguments() {
        let p = Policies::new(&[
            when(
                "homeassistant/*call_action",
                Policy::Ask,
                &[
                    ("/action", "alarm_control_panel.alarm_disarm"),
                    ("/target/entity_id", "alarm_control_panel.64_104058*"),
                ],
            ),
            rule("homeassistant/*", Policy::Allow),
        ])
        .unwrap();
        let disarm = json!({"action": "alarm_control_panel.alarm_disarm",
                            "target": {"entity_id": "alarm_control_panel.64_104058"}});
        assert_eq!(p.policy_for(CALL, &disarm), Policy::Ask);
        // Case does not matter (Home Assistant lowercases both).
        let shouty = json!({"action": "Alarm_Control_Panel.Alarm_Disarm",
                            "target": {"entity_id": "ALARM_CONTROL_PANEL.64_104058"}});
        assert_eq!(p.policy_for(CALL, &shouty), Policy::Ask);
        // Every `when` must match.
        let arm = json!({"action": "alarm_control_panel.alarm_arm_away",
                         "target": {"entity_id": "alarm_control_panel.64_104058"}});
        assert_eq!(p.policy_for(CALL, &arm), Policy::Allow);
        // A pointer that is not there does not match.
        let no_target = json!({"action": "alarm_control_panel.alarm_disarm"});
        assert_eq!(p.policy_for(CALL, &no_target), Policy::Allow);
        assert_eq!(p.policy_for(CALL, &json!({})), Policy::Allow);
        // The tool glob still has to match.
        assert_eq!(p.policy_for("other/call_action", &disarm), Policy::Confirm);
    }

    #[test]
    fn arrays_match_if_any_element_does() {
        let p = Policies::new(&[
            when(
                CALL,
                Policy::Deny,
                &[("/target/entity_id", &format!("*{UNIFI}*"))],
            ),
            rule(CALL, Policy::Allow),
        ])
        .unwrap();
        let both = json!({"target": {"entity_id": ["alarm_control_panel.64_104058", UNIFI]}});
        assert_eq!(p.policy_for(CALL, &both), Policy::Deny);
        let one = json!({"target": {"entity_id": ["alarm_control_panel.64_104058"]}});
        assert_eq!(p.policy_for(CALL, &one), Policy::Allow);
        // Home Assistant also takes a comma-separated string.
        let csv = json!({"target": {"entity_id": format!("light.x, {UNIFI}")}});
        assert_eq!(p.policy_for(CALL, &csv), Policy::Deny);
        // Objects never match.
        let obj = json!({"target": {"entity_id": {"x": UNIFI}}});
        assert_eq!(p.policy_for(CALL, &obj), Policy::Allow);
    }

    #[test]
    fn numbers_and_bools_match_as_json_text() {
        let p = Policies::new(&[
            when("nextcloud/*", Policy::Confirm, &[("/confirm", "true")]),
            when("home/thermostat", Policy::Ask, &[("/temperature", "3?")]),
            when("home/thermostat", Policy::Deny, &[("/steps/1", "1.5")]),
            rule("nextcloud/*", Policy::Allow),
            rule("home/*", Policy::Allow),
        ])
        .unwrap();
        let write = "nextcloud/deck_create_card";
        assert_eq!(
            p.policy_for(write, &json!({"confirm": true})),
            Policy::Confirm
        );
        assert_eq!(
            p.policy_for(write, &json!({"confirm": "true"})),
            Policy::Confirm
        );
        assert_eq!(
            p.policy_for(write, &json!({"confirm": false})),
            Policy::Allow
        );
        assert_eq!(p.policy_for(write, &json!({})), Policy::Allow);
        let t = "home/thermostat";
        assert_eq!(p.policy_for(t, &json!({"temperature": 35})), Policy::Ask);
        assert_eq!(p.policy_for(t, &json!({"temperature": 21})), Policy::Allow);
        assert_eq!(p.policy_for(t, &json!({"steps": [1, 1.5]})), Policy::Deny);
    }

    #[test]
    fn first_match_wins_with_conditions_and_when_any_and_unless() {
        let mut unifi_anywhere = rule(CALL, Policy::Deny);
        unifi_anywhere.when_any = [
            ("/target/entity_id".to_string(), format!("*{UNIFI}*")),
            ("/data/entity_id".to_string(), format!("*{UNIFI}*")),
        ]
        .into();
        let mut only_homesecure = when(CALL, Policy::Deny, &[("/action", "alarm_control_panel.*")]);
        only_homesecure.unless = [("/target/entity_id".to_string(), HOMESECURE.to_string())].into();
        let mut disarm = when(
            CALL,
            Policy::Code,
            &[("/action", "alarm_control_panel.alarm_disarm")],
        );
        disarm.code = Some("alarm".into());
        let p = Policies::new(&[
            unifi_anywhere,
            only_homesecure,
            disarm,
            when(
                CALL,
                Policy::Confirm,
                &[("/action", "alarm_control_panel.*")],
            ),
            rule(CALL, Policy::Ask),
        ])
        .unwrap();
        let call = |action: &str, target: Value| json!({"action": action, "target": target});
        let d = "alarm_control_panel.alarm_disarm";
        // The UniFi panel is refused before the code rule is reached.
        assert_eq!(
            p.policy_for(CALL, &call(d, json!({"entity_id": UNIFI}))),
            Policy::Deny
        );
        assert_eq!(
            p.policy_for(CALL, &json!({"action": d, "data": {"entity_id": UNIFI}})),
            Policy::Deny
        );
        // Anything but the HomeSecure entities is refused, a missing or
        // templated entity included.
        assert_eq!(
            p.policy_for(CALL, &call(d, json!({"area_id": "loft"}))),
            Policy::Deny
        );
        assert_eq!(
            p.policy_for(
                CALL,
                &call(d, json!({"entity_id": "{{ 'alarm_control_panel.x' }}"}))
            ),
            Policy::Deny
        );
        assert_eq!(
            p.policy_for(
                CALL,
                &call(d, json!({"entity_id": ["alarm_control_panel.64_104058", "alarm_control_panel.other"]}))
            ),
            Policy::Deny
        );
        let m = p.matched(
            CALL,
            &call(
                d,
                json!({"entity_id": "alarm_control_panel.64_104058_shed_only"}),
            ),
        );
        assert_eq!(m.policy, Policy::Code);
        assert_eq!(m.code.unwrap().code, "alarm");
        assert_eq!(m.code.unwrap().on_missing, OnMissing::Deny);
        assert!(m.conditional);
        assert_eq!(
            p.policy_for(
                CALL,
                &call(
                    "alarm_control_panel.alarm_arm_away",
                    json!({"entity_id": "alarm_control_panel.64_104058"})
                )
            ),
            Policy::Confirm
        );
        assert_eq!(
            p.policy_for(CALL, &call("light.turn_on", json!({}))),
            Policy::Ask
        );
    }

    #[test]
    fn only_unconditional_denies_hide_a_tool() {
        let p = Policies::new(&[
            when("home/*", Policy::Allow, &[("/name", "kitchen")]),
            rule("home/set_light", Policy::Deny),
            when("home/unlock_door", Policy::Deny, &[("/door", "front")]),
            rule("home/delete_*", Policy::Deny),
            rule("home/*", Policy::Allow),
        ])
        .unwrap();
        // A conditional rule comes first, so some calls may be allowed.
        assert!(!p.always_denies("home/set_light"));
        assert!(!p.always_denies("home/unlock_door"));
        assert!(!p.always_denies("home/other"));
        let p = Policies::new(&[
            when("home/unlock_door", Policy::Deny, &[("/door", "front")]),
            rule("home/delete_*", Policy::Deny),
            rule("home/*", Policy::Allow),
        ])
        .unwrap();
        assert!(p.always_denies("home/delete_all"));
        assert!(!p.always_denies("home/unlock_door"));
        assert_eq!(
            p.policy_for("home/unlock_door", &json!({"door": "front"})),
            Policy::Deny
        );
        assert_eq!(
            p.policy_for("home/unlock_door", &json!({"door": "back"})),
            Policy::Allow
        );
        assert!(!p.always_denies("home/set_light"));
    }

    #[test]
    fn a_code_rule_may_leave_out_the_only_code() {
        let p = Policies::with_codes(&[rule("home/*", Policy::Code)], &["alarm".into()]).unwrap();
        let m = p.matched("home/x", &json!({}));
        assert_eq!(m.code.unwrap().code, "alarm");
        assert!(!m.conditional);
    }
}
