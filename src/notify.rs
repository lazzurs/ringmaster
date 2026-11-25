//! Notifications back to the phone. ntfy today; the trait leaves room for
//! Matrix later.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::json;

use crate::config::NtfyConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// The final answer to a command.
    Result,
    /// Something went wrong (or is delayed).
    Error,
    /// A tool call is waiting for approval.
    Confirm {
        approve_url: Option<String>,
        deny_url: Option<String>,
        view_url: Option<String>,
    },
    /// Informational (e.g. test event received).
    Info,
    /// An operational alert for the operator (database, stalled queue);
    /// `reminder` while a problem lasts.
    Alert { reminder: bool },
    /// An operational problem has cleared.
    Recovered,
}

/// ntfy tag on every operational alert and recovery, to filter on.
pub const ALERT_TAG: &str = "ringmaster-alert";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    pub title: String,
    pub body: String,
    pub kind: Kind,
}

#[async_trait]
pub trait Notifier: Send + Sync {
    async fn send(&self, n: &Notification) -> Result<()>;
}

/// Used when no notifier is configured: logs instead.
pub struct LogNotifier;

#[async_trait]
impl Notifier for LogNotifier {
    async fn send(&self, n: &Notification) -> Result<()> {
        tracing::info!(title = %n.title, body = %n.body, kind = ?n.kind, "notification (no notifier configured)");
        Ok(())
    }
}

pub struct NtfyNotifier {
    http: reqwest::Client,
    url: String,
    topic: String,
    token: Option<String>,
    action_headers: BTreeMap<String, String>,
}

impl NtfyNotifier {
    /// The same server and token, publishing to `topic` (alerts may have
    /// their own topic).
    pub fn with_topic(cfg: &NtfyConfig, topic: &str, token: Option<String>) -> Result<Self> {
        let mut n = Self::new(cfg, token)?;
        n.topic = topic.to_string();
        Ok(n)
    }

    pub fn new(cfg: &NtfyConfig, token: Option<String>) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .user_agent(concat!("ringmaster/", env!("CARGO_PKG_VERSION")))
                .build()?,
            url: cfg.url.trim_end_matches('/').to_string(),
            topic: cfg.topic.clone(),
            token,
            action_headers: cfg.action_headers.clone(),
        })
    }

    /// The JSON body ntfy's publish-as-JSON API takes (`POST /` on the server).
    pub fn payload(&self, n: &Notification) -> serde_json::Value {
        let (priority, tags) = match &n.kind {
            Kind::Result => (3, vec!["white_check_mark"]),
            Kind::Error => (4, vec!["warning"]),
            Kind::Confirm { .. } => (5, vec!["question"]),
            Kind::Info => (2, vec!["information_source"]),
            Kind::Alert { reminder: false } => (5, vec!["rotating_light", ALERT_TAG]),
            Kind::Alert { reminder: true } => (4, vec!["rotating_light", ALERT_TAG]),
            Kind::Recovered => (3, vec!["heavy_check_mark", ALERT_TAG]),
        };
        let mut body = json!({
            "topic": self.topic,
            "title": n.title,
            "message": n.body,
            "priority": priority,
            "tags": tags,
        });
        if let Kind::Confirm {
            approve_url,
            deny_url,
            view_url,
        } = &n.kind
        {
            let mut actions = Vec::new();
            for (label, url) in [("Approve", approve_url), ("Deny", deny_url)] {
                if let Some(url) = url {
                    let mut a = json!({
                        "action": "http",
                        "label": label,
                        "url": url,
                        "method": "POST",
                        "clear": true,
                    });
                    if !self.action_headers.is_empty() {
                        a["headers"] = json!(self.action_headers);
                    }
                    actions.push(a);
                }
            }
            if let Some(url) = view_url {
                actions.push(json!({"action": "view", "label": "Details", "url": url}));
            }
            if !actions.is_empty() {
                body["actions"] = json!(actions);
            }
        }
        body
    }
}

#[async_trait]
impl Notifier for NtfyNotifier {
    async fn send(&self, n: &Notification) -> Result<()> {
        let mut req = self
            .http
            .post(format!("{}/", self.url))
            .json(&self.payload(n));
        if let Some(t) = &self.token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await?;
        let status = resp.status();
        metrics::counter!("ringmaster_notifications_total", "status" => status.as_u16().to_string())
            .increment(1);
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let text: String = text.chars().take(200).collect();
            // In case a proxy echoes the request back.
            let text = crate::config::redact_credentials(&text, self.token.as_deref());
            bail!("ntfy returned {status}: {text}");
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod testing {
    use super::*;
    use std::sync::Mutex;

    /// Collects notifications for assertions.
    #[derive(Default)]
    pub struct Recorder(pub Mutex<Vec<Notification>>);

    #[async_trait]
    impl Notifier for Recorder {
        async fn send(&self, n: &Notification) -> Result<()> {
            self.0.lock().unwrap().push(n.clone());
            Ok(())
        }
    }

    impl Recorder {
        pub fn take(&self) -> Vec<Notification> {
            std::mem::take(&mut self.0.lock().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn publishes_confirm_with_http_actions() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .and(header("authorization", "Bearer tk_test"))
            .and(body_partial_json(json!({
                "topic": "ring",
                "title": "Confirm",
                "priority": 5,
                "actions": [
                    {"action": "http", "label": "Approve", "url": "https://x/confirm/t/approve", "method": "POST", "clear": true,
                     "headers": {"CF-Access-Client-Id": "id"}},
                    {"action": "http", "label": "Deny", "url": "https://x/confirm/t/deny", "method": "POST", "clear": true},
                    {"action": "view", "label": "Details", "url": "https://x/confirm/t"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        let n = NtfyNotifier::new(
            &NtfyConfig {
                url: server.uri(),
                topic: "ring".into(),
                action_headers: [("CF-Access-Client-Id".to_string(), "id".to_string())].into(),
            },
            Some("tk_test".into()),
        )
        .unwrap();
        n.send(&Notification {
            title: "Confirm".into(),
            body: "Unlock the front door?".into(),
            kind: Kind::Confirm {
                approve_url: Some("https://x/confirm/t/approve".into()),
                deny_url: Some("https://x/confirm/t/deny".into()),
                view_url: Some("https://x/confirm/t".into()),
            },
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn alerts_have_their_own_topic_priority_and_tag() {
        let server = MockServer::start().await;
        for (title, priority, tag) in [
            ("a", 5, "rotating_light"),
            ("r", 4, "rotating_light"),
            ("ok", 3, "heavy_check_mark"),
        ] {
            Mock::given(method("POST"))
                .and(body_partial_json(json!({
                    "topic": "ops", "title": title, "priority": priority,
                    "tags": [tag, ALERT_TAG],
                })))
                .respond_with(ResponseTemplate::new(200))
                .expect(1)
                .mount(&server)
                .await;
        }
        let cfg = NtfyConfig {
            url: server.uri(),
            topic: "ring".into(),
            action_headers: Default::default(),
        };
        let n = NtfyNotifier::with_topic(&cfg, "ops", None).unwrap();
        for (title, kind) in [
            ("a", Kind::Alert { reminder: false }),
            ("r", Kind::Alert { reminder: true }),
            ("ok", Kind::Recovered),
        ] {
            n.send(&Notification {
                title: title.into(),
                body: "b".into(),
                kind,
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn reports_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&server)
            .await;
        let n = NtfyNotifier::new(
            &NtfyConfig {
                url: server.uri(),
                topic: "ring".into(),
                action_headers: Default::default(),
            },
            None,
        )
        .unwrap();
        let err = n
            .send(&Notification {
                title: "t".into(),
                body: "b".into(),
                kind: Kind::Result,
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("403"));
    }

    #[tokio::test]
    async fn errors_never_echo_the_token() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(401).set_body_string("bad token: Bearer tk_0123456789"),
            )
            .mount(&server)
            .await;
        let n = NtfyNotifier::new(
            &NtfyConfig {
                url: server.uri(),
                topic: "ring".into(),
                action_headers: Default::default(),
            },
            Some("tk_0123456789".into()),
        )
        .unwrap();
        let err = n
            .send(&Notification {
                title: "t".into(),
                body: "b".into(),
                kind: Kind::Result,
            })
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("401") && !err.contains("tk_0123456789"),
            "{err}"
        );
    }
}
