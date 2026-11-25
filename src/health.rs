//! Operational health: what the service knows about itself, the watchdog
//! that turns it into alerts, and the answers for `/healthz` and `/metrics`.
//!
//! [`Health`] is in-memory state fed by the worker (loop heartbeat, jobs in
//! flight, failures, claim errors) and by blocking work (decodes, audio
//! writes). [`Db`]'s [`DbStatus`](crate::db::DbStatus) records database
//! errors. The [`Watchdog`] is an independent task: every
//! `alerts.check_interval_secs` it checks the heartbeats, the queue and the
//! database (`PRAGMA quick_check` plus a write probe), works out which
//! [`Condition`]s hold, and hands them to the [`Alerter`], which sends one
//! alert when a condition starts, a reminder every `alerts.reminder_secs`
//! while it lasts and one "recovered" message when it clears.
//!
//! Alerts carry counts, job ids, ages and error classes only: never a
//! transcript, a model reply or a secret.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::{AlertsConfig, Config, WORKER_LOOP_MAX_WAIT_SECS};
use crate::db::{Db, DbErrorClass, DbHealth, QueueStats, now};
use crate::notify::{Kind, Notification, Notifier};

/// Job outcomes that count towards the "jobs failing" alert.
const FAILURE_OUTCOMES: &[&str] = &["failed", "expired", "quarantined", "panicked"];

/// Longest a database health check may take.
const DB_CHECK_LIMIT: Duration = Duration::from_secs(30);
/// Longest the watchdog waits for the queue statistics.
const QUEUE_QUERY_LIMIT: Duration = Duration::from_secs(10);
/// Longest one alert may take to send.
const SEND_LIMIT: Duration = Duration::from_secs(20);
/// An alert that could not be sent is retried this often.
const RESEND_SECS: i64 = 60;

/// A problem the watchdog alerts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Condition {
    DatabaseCorrupt,
    DatabaseFailing,
    WorkerStalled,
    QueueStalled,
    JobOverrun,
    JobsFailing,
    BlockingStuck,
}

impl Condition {
    pub const ALL: [Condition; 7] = [
        Condition::DatabaseCorrupt,
        Condition::DatabaseFailing,
        Condition::WorkerStalled,
        Condition::QueueStalled,
        Condition::JobOverrun,
        Condition::JobsFailing,
        Condition::BlockingStuck,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Condition::DatabaseCorrupt => "database_corrupt",
            Condition::DatabaseFailing => "database_failing",
            Condition::WorkerStalled => "worker_stalled",
            Condition::QueueStalled => "queue_stalled",
            Condition::JobOverrun => "job_overrun",
            Condition::JobsFailing => "jobs_failing",
            Condition::BlockingStuck => "blocking_stuck",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Condition::DatabaseCorrupt => "database corrupt",
            Condition::DatabaseFailing => "database failing",
            Condition::WorkerStalled => "worker stalled",
            Condition::QueueStalled => "queue stalled",
            Condition::JobOverrun => "job past its deadline",
            Condition::JobsFailing => "jobs failing",
            Condition::BlockingStuck => "blocking work stuck",
        }
    }
}

/// "45 s", "12 min", "2 h 5 min".
pub fn human(secs: i64) -> String {
    let secs = secs.max(0);
    match secs {
        0..=119 => format!("{secs} s"),
        120..=7199 => format!("{} min", secs / 60),
        _ => format!("{} h {} min", secs / 3600, (secs % 3600) / 60),
    }
}

/// Blocking work in progress (decodes, disk writes), shared with the guards
/// that remove their entry when the thread really finishes.
#[derive(Clone, Default)]
pub struct BlockingTracker {
    entries: Arc<Mutex<HashMap<u64, BlockingEntry>>>,
    next: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct BlockingEntry {
    kind: &'static str,
    started: i64,
    limit_secs: u64,
}

/// Removes its entry when dropped; move it into the blocking closure.
pub struct BlockingGuard {
    entries: Arc<Mutex<HashMap<u64, BlockingEntry>>>,
    id: u64,
}

impl Drop for BlockingGuard {
    fn drop(&mut self) {
        lock(&self.entries).remove(&self.id);
    }
}

impl BlockingTracker {
    /// Register blocking work of `kind` that should be done within
    /// `limit_secs`.
    pub fn start(&self, kind: &'static str, limit_secs: u64) -> BlockingGuard {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.entries).insert(
            id,
            BlockingEntry {
                kind,
                started: now(),
                limit_secs,
            },
        );
        BlockingGuard {
            entries: self.entries.clone(),
            id,
        }
    }

    /// How many are in progress, by kind.
    pub fn count(&self, kind: &str) -> usize {
        lock(&self.entries)
            .values()
            .filter(|e| e.kind == kind)
            .count()
    }

    fn snapshot(&self) -> Vec<BlockingEntry> {
        lock(&self.entries).values().cloned().collect()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// In-memory health state, shared by the worker, the webhook, the watchdog
/// and the HTTP handlers.
pub struct Health {
    cfg: AlertsConfig,
    attempt_timeout_secs: u64,
    worker_beat: AtomicI64,
    watchdog_beat: AtomicI64,
    claim_failures: AtomicU32,
    claim_error: Mutex<Option<DbErrorClass>>,
    /// Job id -> when its task started.
    running: Mutex<HashMap<i64, i64>>,
    /// (when, job id, outcome) of recent failures.
    failures: Mutex<VecDeque<(i64, i64, &'static str)>>,
    pub blocking: BlockingTracker,
    /// Blocking writes the webhook may have running at once.
    pub io_permits: Arc<tokio::sync::Semaphore>,
    queue: Mutex<Option<QueueStats>>,
    active: Mutex<Vec<Condition>>,
}

/// Blocking audio writes the webhook may have running at once.
const IO_PERMITS: usize = 4;

impl Health {
    pub fn new(config: &Config, now: i64) -> Self {
        Self {
            cfg: config.alerts.clone(),
            attempt_timeout_secs: config.agent.attempt_timeout_secs,
            worker_beat: AtomicI64::new(now),
            watchdog_beat: AtomicI64::new(now),
            claim_failures: AtomicU32::new(0),
            claim_error: Mutex::new(None),
            running: Mutex::default(),
            failures: Mutex::default(),
            blocking: BlockingTracker::default(),
            io_permits: Arc::new(tokio::sync::Semaphore::new(IO_PERMITS)),
            queue: Mutex::default(),
            active: Mutex::default(),
        }
    }

    pub fn config(&self) -> &AlertsConfig {
        &self.cfg
    }

    pub fn beat_worker(&self, now: i64) {
        self.worker_beat.store(now, Ordering::Relaxed);
    }

    pub fn beat_watchdog(&self, now: i64) {
        self.watchdog_beat.store(now, Ordering::Relaxed);
    }

    pub fn worker_beat_age(&self, now: i64) -> i64 {
        now - self.worker_beat.load(Ordering::Relaxed)
    }

    pub fn watchdog_beat_age(&self, now: i64) -> i64 {
        now - self.watchdog_beat.load(Ordering::Relaxed)
    }

    /// The watchdog is late after missing three checks (and never sooner
    /// than two minutes, so a slow check is not a stall).
    fn watchdog_stall_secs(&self) -> i64 {
        (3 * self.cfg.check_interval_secs).max(120) as i64
    }

    pub fn job_started(&self, id: i64, now: i64) {
        lock(&self.running).insert(id, now);
    }

    pub fn job_finished(&self, id: i64) {
        lock(&self.running).remove(&id);
    }

    /// Jobs with a task right now.
    pub fn running_jobs(&self) -> Vec<i64> {
        let mut ids: Vec<i64> = lock(&self.running).keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    pub fn record_outcome(&self, job: i64, outcome: &'static str, now: i64) {
        if !FAILURE_OUTCOMES.contains(&outcome) {
            return;
        }
        let mut f = lock(&self.failures);
        f.push_back((now, job, outcome));
        // Keep the window; the count can never grow without bound.
        let window = self.cfg.failure_window_secs as i64;
        while f.front().is_some_and(|(t, ..)| now - t > window) || f.len() > 1000 {
            f.pop_front();
        }
    }

    pub fn claim_ok(&self) {
        self.claim_failures.store(0, Ordering::Relaxed);
    }

    pub fn claim_failed(&self, class: Option<DbErrorClass>) {
        self.claim_failures.fetch_add(1, Ordering::Relaxed);
        *lock(&self.claim_error) = class;
    }

    pub fn set_queue(&self, stats: QueueStats) {
        *lock(&self.queue) = Some(stats);
    }

    /// Conditions active after the last watchdog check.
    pub fn active_alerts(&self) -> Vec<Condition> {
        lock(&self.active).clone()
    }

    /// Why the service is unhealthy, for `/healthz`: only problems that stop
    /// recordings from being stored or processed at all. Slow or failing
    /// jobs, a sleeping Mac or a down MCP server are not the container's
    /// fault and never make it unhealthy.
    pub fn unhealthy(&self, now: i64, db: &DbHealth) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(why) = &db.corrupt {
            out.push(format!("database corrupt: {why}"));
        }
        if let Some(class) = db.storage {
            out.push(format!("database storage: {class}"));
        }
        if db.unresponsive {
            out.push("database unresponsive".into());
        }
        let age = self.worker_beat_age(now);
        if age > self.cfg.worker_stall_secs as i64 {
            out.push(format!("worker loop stalled for {}", human(age)));
        }
        let age = self.watchdog_beat_age(now);
        if age > self.watchdog_stall_secs() {
            out.push(format!("watchdog stalled for {}", human(age)));
        }
        out
    }

    fn db_ok(&self, db: &DbHealth) -> bool {
        db.corrupt.is_none()
            && db.storage.is_none()
            && !db.unresponsive
            && db.consecutive_failures < self.cfg.db_error_threshold
    }

    /// The conditions that hold now, each with the alert text.
    pub fn conditions(&self, now: i64, db: &DbHealth) -> BTreeMap<Condition, String> {
        let cfg = &self.cfg;
        let mut out = BTreeMap::new();
        const NOT_STORED: &str = "New recordings may not be stored or processed; the \
            webhook answers 503 so the phone retries (up to 10 times, a minute apart).";

        if let Some(why) = &db.corrupt {
            out.insert(
                Condition::DatabaseCorrupt,
                format!(
                    "SQLite reports the database is corrupt: {why}. {NOT_STORED} Stop \
                     Ringmaster and restore ringmaster.db from a backup."
                ),
            );
        }

        let mut failing = Vec::new();
        if let Some(class) = db.storage {
            failing.push(format!("Database writes fail: {class}."));
        }
        if db.unresponsive {
            failing.push(format!(
                "The database did not answer a health check within {} s.",
                DB_CHECK_LIMIT.as_secs()
            ));
        }
        // Failures caused by corruption are the corruption alert's business.
        let corruption = |c: Option<DbErrorClass>| c.is_some_and(DbErrorClass::is_corruption);
        if db.consecutive_failures >= cfg.db_error_threshold && !corruption(db.last_error) {
            failing.push(format!(
                "{} database statements failed in a row (last: {}).",
                db.consecutive_failures,
                db.last_error.map_or("unknown".into(), |c| c.to_string())
            ));
        }
        let claims = self.claim_failures.load(Ordering::Relaxed);
        let class = *lock(&self.claim_error);
        if claims >= cfg.db_error_threshold && !corruption(class) {
            failing.push(format!(
                "Claiming queued jobs failed {claims} times in a row (last: {}).",
                class.map_or("unknown".into(), |c| c.to_string())
            ));
        }
        if !failing.is_empty() {
            out.insert(
                Condition::DatabaseFailing,
                format!("{} {NOT_STORED}", failing.join(" ")),
            );
        }

        let age = self.worker_beat_age(now);
        if age > cfg.worker_stall_secs as i64 {
            out.insert(
                Condition::WorkerStalled,
                format!(
                    "The job worker loop has not run for {} (it runs at least every {} s). \
                     Recordings are stored but not processed. Restart Ringmaster if this \
                     lasts.",
                    human(age),
                    WORKER_LOOP_MAX_WAIT_SECS
                ),
            );
        }

        if let Some(q) = lock(&self.queue).as_ref()
            && let Some((id, due)) = q.oldest_due
            && now - due > cfg.queue_stall_secs as i64
        {
            out.insert(
                Condition::QueueStalled,
                format!(
                    "Job {id} has been due for {} without starting ({} due, {} running). \
                     Recordings are stored but not processed.",
                    human(now - due),
                    q.due,
                    q.running
                ),
            );
        }

        let limit = (self.attempt_timeout_secs + cfg.overrun_margin_secs) as i64;
        let mut overrun: Vec<(i64, i64)> = lock(&self.running)
            .iter()
            .filter(|(_, started)| now - **started > limit)
            .map(|(id, started)| (*id, now - started))
            .collect();
        if !overrun.is_empty() {
            overrun.sort_unstable();
            let longest = overrun.iter().map(|(_, a)| *a).max().unwrap_or(0);
            let ids: Vec<String> = overrun.iter().map(|(id, _)| id.to_string()).collect();
            out.insert(
                Condition::JobOverrun,
                format!(
                    "Job {} running for up to {}, past agent.attempt_timeout_secs ({} s) \
                     + {} s: the deadline did not fire. It holds a worker slot; restart \
                     Ringmaster if this lasts (the job is requeued).",
                    ids.join(", "),
                    human(longest),
                    self.attempt_timeout_secs,
                    cfg.overrun_margin_secs
                ),
            );
        }

        {
            let window = cfg.failure_window_secs as i64;
            let f = lock(&self.failures);
            let recent: Vec<&(i64, i64, &str)> =
                f.iter().filter(|(t, ..)| now - t <= window).collect();
            if recent.len() >= cfg.failure_threshold {
                let mut by: BTreeMap<&str, usize> = BTreeMap::new();
                for (_, _, o) in &recent {
                    *by.entry(o).or_default() += 1;
                }
                let breakdown: Vec<String> = by.iter().map(|(o, n)| format!("{o} {n}")).collect();
                let ids: Vec<String> = recent
                    .iter()
                    .rev()
                    .take(5)
                    .map(|(_, id, _)| id.to_string())
                    .collect();
                out.insert(
                    Condition::JobsFailing,
                    format!(
                        "{} jobs failed in the last {} ({}). Latest job ids: {}. See the \
                         audit log.",
                        recent.len(),
                        human(window),
                        breakdown.join(", "),
                        ids.join(", ")
                    ),
                );
            }
        }

        let stuck: Vec<BlockingEntry> = self
            .blocking
            .snapshot()
            .into_iter()
            .filter(|e| now - e.started > (e.limit_secs + cfg.overrun_margin_secs) as i64)
            .collect();
        if !stuck.is_empty() {
            let mut by: BTreeMap<&str, (usize, i64, u64)> = BTreeMap::new();
            for e in &stuck {
                let v = by.entry(e.kind).or_insert((0, 0, e.limit_secs));
                v.0 += 1;
                v.1 = v.1.max(now - e.started);
            }
            let parts: Vec<String> = by
                .iter()
                .map(|(kind, (n, age, limit))| {
                    format!("{n} {kind} for up to {} (limit {limit} s)", human(*age))
                })
                .collect();
            out.insert(
                Condition::BlockingStuck,
                format!(
                    "Blocking work did not return: {}. These threads cannot be stopped; \
                     while the decoders are busy recordings fall back to the phone's \
                     transcription. Restart Ringmaster to free them.",
                    parts.join("; ")
                ),
            );
        }
        out
    }

    /// Set every gauge from the current state. Called after each watchdog
    /// check and on every `/metrics` scrape.
    pub fn export(&self, now: i64, db: &DbHealth) {
        use metrics::gauge;
        gauge!("ringmaster_db_healthy").set(if self.db_ok(db) { 1.0 } else { 0.0 });
        gauge!("ringmaster_worker_heartbeat_age_seconds").set(self.worker_beat_age(now) as f64);
        gauge!("ringmaster_watchdog_heartbeat_age_seconds").set(self.watchdog_beat_age(now) as f64);
        gauge!("ringmaster_jobs_running").set(lock(&self.running).len() as f64);
        if let Some(q) = lock(&self.queue).as_ref() {
            gauge!("ringmaster_jobs_due").set(q.due as f64);
            gauge!("ringmaster_oldest_due_job_age_seconds")
                .set(q.oldest_due.map_or(0, |(_, due)| (now - due).max(0)) as f64);
        }
        let active = self.active_alerts();
        gauge!("ringmaster_alerts_active").set(active.len() as f64);
        for c in Condition::ALL {
            gauge!("ringmaster_alert_active", "condition" => c.key()).set(if active.contains(&c) {
                1.0
            } else {
                0.0
            });
        }
        for kind in ["decode", "audio_write"] {
            gauge!("ringmaster_blocking_tasks", "kind" => kind)
                .set(self.blocking.count(kind) as f64);
        }
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            gauge!("ringmaster_tokio_alive_tasks").set(rt.metrics().num_alive_tasks() as f64);
        }
    }
}

// ------------------------------------------------------------------ alerts

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Active {
    since: i64,
    /// Last time an alert for it was sent (or tried).
    last_sent: i64,
    /// The first alert went out, so a "recovered" message makes sense.
    delivered: bool,
}

/// Debounced alert state. Kept in `alerts.json` in the data directory so a
/// restart (the usual fix) still ends with a "recovered" message and a
/// crash loop does not repeat the alert.
pub struct AlertBook {
    active: BTreeMap<Condition, Active>,
    file: Option<PathBuf>,
}

/// One message the book wants sent.
struct Planned {
    condition: Condition,
    note: Notification,
    first: bool,
}

impl AlertBook {
    pub fn load(file: Option<PathBuf>) -> Self {
        let active = file
            .as_ref()
            .and_then(|f| std::fs::read(f).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Self { active, file }
    }

    pub fn active(&self) -> Vec<Condition> {
        self.active.keys().copied().collect()
    }

    fn plan(
        &mut self,
        now: i64,
        current: &BTreeMap<Condition, String>,
        reminder_secs: u64,
    ) -> Vec<Planned> {
        let mut out = Vec::new();
        for (&c, detail) in current {
            let alert = |reminder: bool, since: i64| Notification {
                title: if reminder {
                    format!("Ringmaster alert (ongoing): {}", c.title())
                } else {
                    format!("Ringmaster alert: {}", c.title())
                },
                body: if reminder {
                    format!("{detail} Ongoing for {}.", human(now - since))
                } else {
                    detail.clone()
                },
                kind: Kind::Alert { reminder },
            };
            match self.active.get_mut(&c) {
                None => {
                    self.active.insert(
                        c,
                        Active {
                            since: now,
                            last_sent: now,
                            delivered: false,
                        },
                    );
                    out.push(Planned {
                        condition: c,
                        note: alert(false, now),
                        first: true,
                    });
                }
                Some(a) if !a.delivered && now - a.last_sent >= RESEND_SECS => {
                    a.last_sent = now;
                    out.push(Planned {
                        condition: c,
                        note: alert(false, a.since),
                        first: true,
                    });
                }
                Some(a)
                    if a.delivered
                        && reminder_secs > 0
                        && now - a.last_sent >= reminder_secs as i64 =>
                {
                    a.last_sent = now;
                    out.push(Planned {
                        condition: c,
                        note: alert(true, a.since),
                        first: false,
                    });
                }
                Some(_) => {}
            }
        }
        let cleared: Vec<Condition> = self
            .active
            .keys()
            .filter(|c| !current.contains_key(c))
            .copied()
            .collect();
        for c in cleared {
            let a = self.active.remove(&c).expect("listed above");
            if a.delivered {
                out.push(Planned {
                    condition: c,
                    note: Notification {
                        title: format!("Ringmaster recovered: {}", c.title()),
                        body: format!("Cleared after {}.", human(now - a.since)),
                        kind: Kind::Recovered,
                    },
                    first: false,
                });
            } else {
                tracing::info!(
                    alert = c.key(),
                    "condition cleared before its alert was sent"
                );
            }
        }
        out
    }

    fn mark_delivered(&mut self, c: Condition) {
        if let Some(a) = self.active.get_mut(&c) {
            a.delivered = true;
        }
    }

    fn save(&self) {
        let Some(file) = &self.file else { return };
        let write = || -> std::io::Result<()> {
            let tmp = file.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_vec(&self.active)?)?;
            std::fs::rename(&tmp, file)
        };
        // The data directory may be the thing that is broken; alerting goes on.
        if let Err(e) = write() {
            tracing::warn!(error = %e, file = %file.display(), "could not save alert state");
        }
    }
}

/// Sends alerts through the notifier, debounced by an [`AlertBook`]. A
/// notifier that fails or hangs is logged and retried later; it never blocks
/// the caller for more than [`SEND_LIMIT`].
pub struct Alerter {
    notifier: Arc<dyn Notifier>,
    enabled: bool,
    reminder_secs: u64,
    book: tokio::sync::Mutex<AlertBook>,
}

impl Alerter {
    pub fn new(notifier: Arc<dyn Notifier>, cfg: &AlertsConfig, book: AlertBook) -> Self {
        Self {
            notifier,
            enabled: cfg.enabled,
            reminder_secs: cfg.reminder_secs,
            book: tokio::sync::Mutex::new(book),
        }
    }

    /// Bring the alerts in line with the conditions that hold now. Returns
    /// the active conditions.
    pub async fn apply(&self, now: i64, current: &BTreeMap<Condition, String>) -> Vec<Condition> {
        let mut book = self.book.lock().await;
        let before: Vec<(Condition, bool)> =
            book.active.iter().map(|(c, a)| (*c, a.delivered)).collect();
        let planned = book.plan(now, current, self.reminder_secs);
        for p in &planned {
            match p.note.kind {
                Kind::Recovered => {
                    tracing::warn!(alert = p.condition.key(), title = %p.note.title, body = %p.note.body, "alert cleared")
                }
                _ => {
                    tracing::error!(alert = p.condition.key(), title = %p.note.title, body = %p.note.body, "alert")
                }
            }
        }
        let sends = planned.iter().map(|p| self.send(&p.note));
        let results = futures_util::future::join_all(sends).await;
        for (p, ok) in planned.iter().zip(results) {
            if ok && p.first {
                book.mark_delivered(p.condition);
            }
        }
        let after: Vec<(Condition, bool)> =
            book.active.iter().map(|(c, a)| (*c, a.delivered)).collect();
        if before != after {
            book.save();
        }
        book.active()
    }

    async fn send(&self, n: &Notification) -> bool {
        if !self.enabled {
            return true;
        }
        match tokio::time::timeout(SEND_LIMIT, self.notifier.send(n)).await {
            Ok(Ok(())) => true,
            Ok(Err(e)) => {
                tracing::error!(error = %e, title = %n.title, "could not send alert; it is logged above and will be retried");
                false
            }
            Err(_) => {
                tracing::error!(title = %n.title, "sending an alert timed out; it is logged above and will be retried");
                false
            }
        }
    }
}

// ---------------------------------------------------------------- watchdog

/// The independent task that checks everything and alerts.
pub struct Watchdog {
    health: Arc<Health>,
    db: Db,
    alerter: Alerter,
    last_db_check: AtomicI64,
    metrics: Option<metrics_exporter_prometheus::PrometheusHandle>,
}

impl Watchdog {
    pub fn new(health: Arc<Health>, db: Db, alerter: Alerter) -> Self {
        Self {
            health,
            db,
            alerter,
            last_db_check: AtomicI64::new(i64::MIN),
            metrics: None,
        }
    }

    /// Also run the Prometheus recorder's upkeep on every check, so
    /// histogram samples do not pile up in memory when nobody scrapes.
    pub fn with_metrics(
        mut self,
        handle: Option<metrics_exporter_prometheus::PrometheusHandle>,
    ) -> Self {
        self.metrics = handle;
        self
    }

    /// One check: heartbeats, database, queue; then alerts and gauges.
    /// Every step is bounded, so a hung database cannot stall it.
    pub async fn check(&self, now: i64) {
        self.health.beat_watchdog(now);
        let cfg = self.health.config().clone();
        let status = self.db.status().clone();

        let snapshot = status.snapshot();
        let suspect = snapshot.corrupt.is_some()
            || snapshot.storage.is_some()
            || snapshot.unresponsive
            || snapshot.consecutive_failures > 0;
        let last = self.last_db_check.load(Ordering::Relaxed);
        if suspect || now.saturating_sub(last) >= cfg.db_check_interval_secs as i64 {
            self.last_db_check.store(now, Ordering::Relaxed);
            let outcome = self.db.health_check(now, DB_CHECK_LIMIT).await;
            tracing::debug!(?outcome, "database health check");
        }

        match tokio::time::timeout(QUEUE_QUERY_LIMIT, self.db.queue_stats(now)).await {
            Ok(Ok(stats)) => self.health.set_queue(stats),
            // Already classified and recorded by the database layer.
            Ok(Err(e)) => tracing::warn!(error = %e, "reading queue statistics failed"),
            Err(_) => status.record_error(DbErrorClass::Unresponsive),
        }

        let db = status.snapshot();
        let current = self.health.conditions(now, &db);
        let active = self.alerter.apply(now, &current).await;
        *lock(&self.health.active) = active;
        self.health.export(now, &db);
        if let Some(m) = &self.metrics {
            m.run_upkeep();
        }
    }

    pub async fn run(self: Arc<Self>, shutdown: CancellationToken) {
        let interval = Duration::from_secs(self.health.config().check_interval_secs);
        loop {
            self.check(now()).await;
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                // A corruption or storage error: check (and alert) now, but
                // not more than once a second.
                _ = self.db.status().wake.notified() => {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(1)) => {}
                        _ = shutdown.cancelled() => break,
                    }
                }
                _ = shutdown.cancelled() => break,
            }
        }
        tracing::info!("watchdog stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notify::testing::Recorder;

    fn health(extra: &str) -> Health {
        let cfg = Config::from_toml(extra).unwrap();
        Health::new(&cfg, 1000)
    }

    #[test]
    fn durations_read_well() {
        assert_eq!(human(45), "45 s");
        assert_eq!(human(720), "12 min");
        assert_eq!(human(7500), "2 h 5 min");
    }

    #[test]
    fn stalls_overruns_and_failure_bursts_are_detected() {
        let h = health("[agent]\nattempt_timeout_secs = 100\n");
        let db = DbHealth::default();
        assert!(h.conditions(1000, &db).is_empty());

        // The worker loop stopped beating.
        let c = h.conditions(1000 + 121, &db);
        assert!(c[&Condition::WorkerStalled].contains("2 min"), "{c:?}");
        assert!(h.unhealthy(1000 + 121, &db)[0].contains("worker loop stalled"));
        h.beat_worker(1121);
        assert!(
            !h.conditions(1121, &db)
                .contains_key(&Condition::WorkerStalled)
        );

        // A job past its deadline + margin.
        h.job_started(7, 1000);
        assert!(!h.conditions(1150, &db).contains_key(&Condition::JobOverrun));
        h.beat_worker(1161);
        let c = h.conditions(1161, &db);
        assert!(c[&Condition::JobOverrun].starts_with("Job 7 "), "{c:?}");
        h.job_finished(7);
        assert!(!h.conditions(1161, &db).contains_key(&Condition::JobOverrun));

        // Three failures in the window; refusals do not count.
        for (i, o) in ["failed", "refused", "quarantined", "panicked"]
            .iter()
            .enumerate()
        {
            h.record_outcome(i as i64 + 1, o, 1161);
        }
        let c = h.conditions(1161, &db);
        let text = &c[&Condition::JobsFailing];
        assert!(text.starts_with("3 jobs failed"), "{text}");
        assert!(text.contains("Latest job ids: 4, 3, 1"), "{text}");
        assert!(
            !h.conditions(1161 + 901, &db)
                .contains_key(&Condition::JobsFailing)
        );

        // The queue: a job due for longer than queue_stall_secs.
        h.set_queue(QueueStats {
            oldest_due: Some((9, 1000)),
            due: 1,
            running: 0,
        });
        h.beat_worker(1700);
        let c = h.conditions(1700, &db);
        assert!(c[&Condition::QueueStalled].starts_with("Job 9 has been due for 11 min"));
    }

    #[test]
    fn blocking_work_past_its_limit_is_reported_until_it_returns() {
        let h = health("");
        let guard = h.blocking.start("decode", 10);
        assert_eq!(h.blocking.count("decode"), 1);
        let start = now();
        assert!(
            !h.conditions(start, &DbHealth::default())
                .contains_key(&Condition::BlockingStuck)
        );
        h.beat_worker(start + 100);
        let c = h.conditions(start + 100, &DbHealth::default());
        assert!(c[&Condition::BlockingStuck].contains("1 decode"), "{c:?}");
        drop(guard);
        assert_eq!(h.blocking.count("decode"), 0);
    }

    struct Failing;

    #[async_trait::async_trait]
    impl Notifier for Failing {
        async fn send(&self, _: &Notification) -> anyhow::Result<()> {
            anyhow::bail!("ntfy is down")
        }
    }

    fn cond(c: Condition) -> BTreeMap<Condition, String> {
        [(c, "detail.".to_string())].into()
    }

    #[tokio::test]
    async fn alerts_are_debounced_reminded_and_recovered_once() {
        let rec = Arc::new(Recorder::default());
        let cfg = AlertsConfig {
            reminder_secs: 600,
            ..Default::default()
        };
        let a = Alerter::new(rec.clone(), &cfg, AlertBook::load(None));
        let on = cond(Condition::WorkerStalled);
        a.apply(0, &on).await;
        a.apply(30, &on).await;
        a.apply(60, &on).await;
        let sent = rec.take();
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert_eq!(sent[0].title, "Ringmaster alert: worker stalled");
        assert_eq!(sent[0].kind, Kind::Alert { reminder: false });
        a.apply(600, &on).await;
        let sent = rec.take();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].kind, Kind::Alert { reminder: true });
        assert!(
            sent[0].body.ends_with("Ongoing for 10 min."),
            "{}",
            sent[0].body
        );
        a.apply(630, &BTreeMap::new()).await;
        a.apply(660, &BTreeMap::new()).await;
        let sent = rec.take();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].title, "Ringmaster recovered: worker stalled");
        assert_eq!(sent[0].kind, Kind::Recovered);
    }

    #[tokio::test]
    async fn a_down_notifier_never_blocks_and_the_alert_is_retried() {
        let cfg = AlertsConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("alerts.json");
        let a = Alerter::new(Arc::new(Failing), &cfg, AlertBook::load(Some(file.clone())));
        let on = cond(Condition::DatabaseCorrupt);
        assert_eq!(a.apply(0, &on).await, vec![Condition::DatabaseCorrupt]);

        // ntfy is back: the alert goes out on the next try, once.
        let rec = Arc::new(Recorder::default());
        let a = Alerter::new(rec.clone(), &cfg, AlertBook::load(Some(file.clone())));
        a.apply(30, &on).await;
        assert!(rec.take().is_empty(), "retried at most once a minute");
        a.apply(60, &on).await;
        a.apply(90, &on).await;
        assert_eq!(rec.take().len(), 1);

        // A restart (new process, same file) that fixed it says so once.
        let a = Alerter::new(rec.clone(), &cfg, AlertBook::load(Some(file)));
        a.apply(120, &BTreeMap::new()).await;
        a.apply(150, &BTreeMap::new()).await;
        let sent = rec.take();
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert_eq!(sent[0].title, "Ringmaster recovered: database corrupt");
    }
}
