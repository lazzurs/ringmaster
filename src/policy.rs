//! Tool safety policies and per-profile tool allow-lists.
//!
//! Tools are named `<server>/<tool>` for matching. Policy rules are checked in
//! order and the first match wins; a tool no rule matches needs confirmation.

use anyhow::Result;
use globset::{Glob, GlobMatcher, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};

use crate::config::ToolPolicyRule;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    /// Run without asking.
    Allow,
    /// Pause the job and ask on the phone first.
    Confirm,
    /// Never run; the tool is not even offered to Claude.
    Deny,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Allow => "allow",
            Policy::Confirm => "confirm",
            Policy::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Policies {
    rules: Vec<(GlobMatcher, Policy)>,
}

impl Policies {
    pub fn new(rules: &[ToolPolicyRule]) -> Result<Self> {
        let rules = rules
            .iter()
            .map(|r| Ok((glob(&r.pattern)?.compile_matcher(), r.policy)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { rules })
    }

    /// Policy for `<server>/<tool>`; unknown tools default to `confirm`.
    pub fn policy_for(&self, qualified: &str) -> Policy {
        self.rules
            .iter()
            .find(|(m, _)| m.is_match(qualified))
            .map(|(_, p)| *p)
            .unwrap_or(Policy::Confirm)
    }
}

/// Which tools a profile may see.
#[derive(Debug, Clone)]
pub struct ToolFilter {
    set: GlobSet,
    /// Server names the profile can reach, `None` meaning any.
    servers: Option<Vec<String>>,
    /// Servers named literally by at least one glob (`nextcloud/...`). A job
    /// cannot run without them; servers reached only through a wildcard
    /// (`*/*`) are optional and are skipped while they are down.
    named: Vec<String>,
}

impl ToolFilter {
    pub fn new(globs: &[String]) -> Result<Self> {
        let mut b = GlobSetBuilder::new();
        let mut servers = Some(Vec::new());
        let mut named: Vec<String> = Vec::new();
        for g in globs {
            b.add(glob(g)?);
            let server = g.split('/').next().unwrap_or_default();
            let literal = !server.is_empty() && !server.contains(['*', '?', '[', '{']);
            if literal && !named.iter().any(|n| n == server) {
                named.push(server.to_string());
            }
            match (&mut servers, literal) {
                (Some(list), true) => list.push(server.to_string()),
                _ => servers = None,
            }
        }
        Ok(Self {
            set: b.build()?,
            servers,
            named,
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

    /// Whether the profile names `server` explicitly, so a job must wait for
    /// it rather than run without its tools.
    pub fn requires_server(&self, server: &str) -> bool {
        self.named.iter().any(|s| s == server)
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

    fn rule(p: &str, policy: Policy) -> ToolPolicyRule {
        ToolPolicyRule {
            pattern: p.into(),
            policy,
        }
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
        assert_eq!(p.policy_for("homey/unlock_door"), Policy::Confirm);
        assert_eq!(p.policy_for("homey/set_light"), Policy::Allow);
        assert_eq!(p.policy_for("homey/delete_flow"), Policy::Deny);
        assert_eq!(p.policy_for("nextcloud/deck_delete_card"), Policy::Deny);
        assert_eq!(p.policy_for("nextcloud/deck_get_boards"), Policy::Allow);
        assert_eq!(p.policy_for("nextcloud/deck_create_card"), Policy::Confirm);
    }

    #[test]
    fn star_does_not_cross_server_separator() {
        let p = Policies::new(&[rule("homey*", Policy::Allow)]).unwrap();
        assert_eq!(p.policy_for("homey/set_light"), Policy::Confirm);
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

    #[test]
    fn tool_filter_required_servers() {
        let f = ToolFilter::new(&["nextcloud/deck_*".into()]).unwrap();
        assert!(f.requires_server("nextcloud"));
        assert!(!f.requires_server("homey"));

        // Wildcard server: every server is usable, none is required.
        let all = ToolFilter::new(&["*/*".into()]).unwrap();
        assert!(!all.requires_server("nextcloud"));

        // Mixed: the named one is required, the rest optional.
        let mixed = ToolFilter::new(&["nextcloud/deck_*".into(), "*/get_*".into()]).unwrap();
        assert!(mixed.requires_server("nextcloud"));
        assert!(!mixed.requires_server("homey"));
        assert!(mixed.may_use_server("homey"));
    }
}
