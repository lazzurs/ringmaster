//! SQLite persistence: deliveries (dedupe), the durable job queue,
//! confirmations and the audit log.
//!
//! One connection behind a mutex, used from `spawn_blocking`. Ringmaster sees a
//! handful of requests a day, so there is nothing to gain from a pool.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde_json::Value;

use crate::agent::AgentState;

const SCHEMA_VERSION: i64 = 1;

const MIGRATION_1: &str = r#"
CREATE TABLE deliveries (
    delivery_id         TEXT PRIMARY KEY,
    received_at         INTEGER NOT NULL,
    trigger             TEXT NOT NULL,
    is_test             INTEGER NOT NULL,
    recorded_at_ms      INTEGER,
    client              TEXT,
    audio_path          TEXT,
    audio_bytes         INTEGER,
    phone_transcription TEXT,
    profile             TEXT,
    outcome             TEXT NOT NULL
);
CREATE INDEX deliveries_received ON deliveries(received_at);

CREATE TABLE jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_id       TEXT NOT NULL UNIQUE REFERENCES deliveries(delivery_id),
    profile           TEXT NOT NULL,
    status            TEXT NOT NULL,
    attempts          INTEGER NOT NULL DEFAULT 0,
    next_attempt_at   INTEGER NOT NULL,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    transcript        TEXT,
    transcript_source TEXT,
    agent_state       TEXT,
    result_text       TEXT,
    last_error        TEXT,
    retry_notified    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX jobs_ready ON jobs(status, next_attempt_at);

CREATE TABLE confirmations (
    token_hash  TEXT PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id),
    summary     TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    decision    TEXT,
    decided_at  INTEGER
);
CREATE INDEX confirmations_job ON confirmations(job_id);

CREATE TABLE audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    job_id      INTEGER,
    delivery_id TEXT,
    kind        TEXT NOT NULL,
    detail      TEXT NOT NULL
);
CREATE INDEX audit_job ON audit(job_id);
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    AwaitingConfirmation,
    Done,
    Failed,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::AwaitingConfirmation => "awaiting_confirmation",
            JobStatus::Done => "done",
            JobStatus::Failed => "failed",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "running" => JobStatus::Running,
            "awaiting_confirmation" => JobStatus::AwaitingConfirmation,
            "done" => JobStatus::Done,
            "failed" => JobStatus::Failed,
            _ => JobStatus::Queued,
        }
    }
}

/// A new delivery as accepted by the webhook.
#[derive(Debug, Clone)]
pub struct NewDelivery {
    pub delivery_id: String,
    pub received_at: i64,
    pub trigger: String,
    pub is_test: bool,
    pub recorded_at_ms: Option<i64>,
    pub client: Option<String>,
    pub audio_path: Option<String>,
    pub audio_bytes: Option<i64>,
    pub phone_transcription: Option<String>,
    pub profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome {
    Duplicate,
    /// Stored without a job (test events).
    Recorded,
    Queued {
        job_id: i64,
    },
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: i64,
    pub delivery_id: String,
    pub profile: String,
    pub status: JobStatus,
    pub attempts: i64,
    pub created_at: i64,
    pub transcript: Option<String>,
    pub transcript_source: Option<String>,
    pub agent_state: Option<AgentState>,
    pub retry_notified: bool,
    // From the delivery.
    pub trigger: String,
    pub recorded_at_ms: Option<i64>,
    pub audio_path: Option<String>,
    pub phone_transcription: Option<String>,
}

/// What [`Db::claim_next_job`] took off the queue.
#[derive(Debug)]
pub enum Claim {
    /// A job, now marked `running`.
    Ready(Box<Job>),
    /// A job whose stored data no longer loads. It has been marked `failed`
    /// (and audited as `job_quarantined`) so it is never claimed again.
    Quarantined(Quarantined),
}

/// What could be salvaged from a job that no longer loads.
#[derive(Debug, Clone)]
pub struct Quarantined {
    pub id: i64,
    pub delivery_id: Option<String>,
    /// The job's transcript, else the phone's transcription, if readable.
    pub heard: Option<String>,
    pub error: String,
}

#[derive(Debug, Clone)]
pub struct Confirmation {
    pub job_id: i64,
    pub summary: String,
    pub expires_at: i64,
    pub decision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecideOutcome {
    Recorded { job_id: i64 },
    AlreadyDecided(String),
    Expired,
    Unknown,
}

#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("opening database {}", path.display()))?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        // WAL for concurrent readers; FULL sync because a 2xx tells the phone
        // it can forget the recording.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version < 1 {
            conn.execute_batch(MIGRATION_1)?;
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    async fn call<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
            f(&mut guard)
        })
        .await
        .context("database task panicked")?
    }

    pub async fn ping(&self) -> Result<()> {
        self.call(|c| {
            c.query_row("SELECT 1", [], |_| Ok(()))?;
            Ok(())
        })
        .await
    }

    pub async fn delivery_exists(&self, delivery_id: &str) -> Result<bool> {
        let id = delivery_id.to_string();
        self.call(move |c| {
            Ok(c.query_row(
                "SELECT 1 FROM deliveries WHERE delivery_id = ?1",
                [id],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
        })
        .await
    }

    /// Insert a delivery and, when `queue` is set, its job, atomically.
    /// A delivery id seen before is reported as a duplicate and not touched.
    pub async fn insert_delivery(&self, d: NewDelivery, queue: bool) -> Result<InsertOutcome> {
        self.call(move |c| {
            let tx = c.transaction()?;
            let outcome = if queue { "queued" } else { "recorded" };
            let inserted = tx.execute(
                "INSERT OR IGNORE INTO deliveries (delivery_id, received_at, trigger, is_test, \
                 recorded_at_ms, client, audio_path, audio_bytes, phone_transcription, profile, outcome) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    d.delivery_id,
                    d.received_at,
                    d.trigger,
                    d.is_test,
                    d.recorded_at_ms,
                    d.client,
                    d.audio_path,
                    d.audio_bytes,
                    d.phone_transcription,
                    d.profile,
                    outcome
                ],
            )?;
            if inserted == 0 {
                return Ok(InsertOutcome::Duplicate);
            }
            let result = if queue {
                let profile = d.profile.clone().context("queued delivery needs a profile")?;
                tx.execute(
                    "INSERT INTO jobs (delivery_id, profile, status, next_attempt_at, created_at, updated_at) \
                     VALUES (?1, ?2, 'queued', ?3, ?3, ?3)",
                    params![d.delivery_id, profile, d.received_at],
                )?;
                InsertOutcome::Queued {
                    job_id: tx.last_insert_rowid(),
                }
            } else {
                InsertOutcome::Recorded
            };
            tx.commit()?;
            Ok(result)
        })
        .await
    }

    /// Jobs left `running` by a crash, a restart or a lost worker task go
    /// back to the queue. `in_flight` are the jobs this process is working on
    /// right now; they are left alone.
    pub async fn requeue_running(&self, now: i64, in_flight: &[i64]) -> Result<usize> {
        let in_flight = serde_json::to_string(in_flight)?;
        self.call(move |c| {
            Ok(c.execute(
                "UPDATE jobs SET status = 'queued', next_attempt_at = ?1, updated_at = ?1 \
                 WHERE status = 'running' AND id NOT IN (SELECT value FROM json_each(?2))",
                params![now, in_flight],
            )?)
        })
        .await
    }

    /// Claim the oldest ready job that is not in `in_flight`, marking it
    /// running. Claiming is one `UPDATE ... RETURNING` in an immediate
    /// transaction, so two workers (even on separate connections) never take
    /// the same job.
    ///
    /// A job whose stored data no longer loads (an `AgentState` from an
    /// incompatible version, a corrupted column, a missing delivery) is
    /// quarantined in the same transaction instead of being left at the head
    /// of the queue, where it would block every job behind it.
    pub async fn claim_next_job(&self, now: i64, in_flight: &[i64]) -> Result<Option<Claim>> {
        let in_flight = serde_json::to_string(in_flight)?;
        self.call(move |c| {
            let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let id: Option<i64> = tx
                .query_row(
                    "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ?1 \
                     WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND next_attempt_at <= ?1 \
                                 AND id NOT IN (SELECT value FROM json_each(?2)) \
                                 ORDER BY next_attempt_at, id LIMIT 1) \
                     AND status = 'queued' RETURNING id",
                    params![now, in_flight],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(id) = id else {
                return Ok(None);
            };
            let reason = match load_job(&tx, id) {
                Ok(Some(job)) => {
                    tx.commit()?;
                    return Ok(Some(Claim::Ready(Box::new(job))));
                }
                Ok(None) => "its delivery record is missing".to_string(),
                Err(LoadError::Unreadable(why)) => why,
                Err(LoadError::Db(e)) => return Err(e),
            };
            let q = quarantine(&tx, id, now, reason)?;
            tx.commit()?;
            Ok(Some(Claim::Quarantined(q)))
        })
        .await
    }

    pub async fn get_job(&self, id: i64) -> Result<Option<Job>> {
        self.call(move |c| {
            load_job(c, id).map_err(|e| match e {
                LoadError::Unreadable(why) => anyhow::anyhow!(why),
                LoadError::Db(e) => e,
            })
        })
        .await
    }

    pub async fn save_transcript(&self, id: i64, text: String, source: String) -> Result<()> {
        self.call(move |c| {
            c.execute(
                "UPDATE jobs SET transcript = ?2, transcript_source = ?3, updated_at = ?4 WHERE id = ?1",
                params![id, text, source, now()],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn save_agent_state(&self, id: i64, state: &AgentState) -> Result<()> {
        let json = serde_json::to_string(state)?;
        self.call(move |c| {
            c.execute(
                "UPDATE jobs SET agent_state = ?2, updated_at = ?3 WHERE id = ?1",
                params![id, json, now()],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn set_status(&self, id: i64, status: JobStatus) -> Result<()> {
        self.call(move |c| {
            c.execute(
                "UPDATE jobs SET status = ?2, updated_at = ?3 WHERE id = ?1",
                params![id, status.as_str(), now()],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn finish_job(
        &self,
        id: i64,
        status: JobStatus,
        result_text: Option<String>,
        error: Option<String>,
    ) -> Result<()> {
        self.call(move |c| {
            c.execute(
                "UPDATE jobs SET status = ?2, result_text = ?3, last_error = COALESCE(?4, last_error), \
                 updated_at = ?5 WHERE id = ?1",
                params![id, status.as_str(), result_text, error, now()],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn reschedule(&self, id: i64, next_attempt_at: i64, error: String) -> Result<()> {
        self.call(move |c| {
            c.execute(
                "UPDATE jobs SET status = 'queued', next_attempt_at = ?2, last_error = ?3, updated_at = ?4 \
                 WHERE id = ?1",
                params![id, next_attempt_at, error, now()],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn mark_retry_notified(&self, id: i64) -> Result<()> {
        self.call(move |c| {
            c.execute("UPDATE jobs SET retry_notified = 1 WHERE id = ?1", [id])?;
            Ok(())
        })
        .await
    }

    pub async fn queue_depth(&self) -> Result<i64> {
        self.call(|c| {
            Ok(c.query_row(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')",
                [],
                |r| r.get(0),
            )?)
        })
        .await
    }

    pub async fn next_due(&self) -> Result<Option<i64>> {
        self.call(|c| {
            Ok(c.query_row(
                "SELECT MIN(next_attempt_at) FROM jobs WHERE status = 'queued'",
                [],
                |r| r.get(0),
            )?)
        })
        .await
    }

    // ---- confirmations ----

    pub async fn create_confirmation(
        &self,
        token_hash: String,
        job_id: i64,
        summary: String,
        expires_at: i64,
    ) -> Result<()> {
        self.call(move |c| {
            c.execute(
                "INSERT INTO confirmations (token_hash, job_id, summary, created_at, expires_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![token_hash, job_id, summary, now(), expires_at],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn get_confirmation(&self, token_hash: String) -> Result<Option<Confirmation>> {
        self.call(move |c| {
            Ok(c.query_row(
                "SELECT job_id, summary, expires_at, decision FROM confirmations WHERE token_hash = ?1",
                [token_hash],
                |r| {
                    Ok(Confirmation {
                        job_id: r.get(0)?,
                        summary: r.get(1)?,
                        expires_at: r.get(2)?,
                        decision: r.get(3)?,
                    })
                },
            )
            .optional()?)
        })
        .await
    }

    /// Record a decision once. The job goes back to the queue to resume.
    pub async fn decide(
        &self,
        token_hash: String,
        decision: String,
        now: i64,
    ) -> Result<DecideOutcome> {
        self.call(move |c| {
            let tx = c.transaction()?;
            let row: Option<(i64, i64, Option<String>)> = tx
                .query_row(
                    "SELECT job_id, expires_at, decision FROM confirmations WHERE token_hash = ?1",
                    [&token_hash],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .optional()?;
            let outcome = match row {
                None => DecideOutcome::Unknown,
                Some((_, _, Some(d))) => DecideOutcome::AlreadyDecided(d),
                Some((_, exp, None)) if exp <= now => DecideOutcome::Expired,
                Some((job_id, _, None)) => {
                    tx.execute(
                        "UPDATE confirmations SET decision = ?2, decided_at = ?3 \
                         WHERE token_hash = ?1 AND decision IS NULL",
                        params![token_hash, decision, now],
                    )?;
                    tx.execute(
                        "UPDATE jobs SET status = 'queued', next_attempt_at = ?2, updated_at = ?2 \
                         WHERE id = ?1 AND status = 'awaiting_confirmation'",
                        params![job_id, now],
                    )?;
                    DecideOutcome::Recorded { job_id }
                }
            };
            tx.commit()?;
            Ok(outcome)
        })
        .await
    }

    /// Mark undecided, expired confirmations and requeue their jobs. Returns
    /// the affected job ids.
    pub async fn expire_confirmations(&self, now: i64) -> Result<Vec<i64>> {
        self.call(move |c| {
            let tx = c.transaction()?;
            let ids: Vec<i64> = {
                let mut st = tx.prepare(
                    "SELECT job_id FROM confirmations WHERE decision IS NULL AND expires_at <= ?1",
                )?;
                st.query_map([now], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            tx.execute(
                "UPDATE confirmations SET decision = 'expired', decided_at = ?1 \
                 WHERE decision IS NULL AND expires_at <= ?1",
                [now],
            )?;
            for id in &ids {
                tx.execute(
                    "UPDATE jobs SET status = 'queued', next_attempt_at = ?2, updated_at = ?2 \
                     WHERE id = ?1 AND status = 'awaiting_confirmation'",
                    params![id, now],
                )?;
            }
            tx.commit()?;
            Ok(ids)
        })
        .await
    }

    /// The latest decision for a job's confirmation, if any.
    pub async fn latest_decision(&self, job_id: i64) -> Result<Option<String>> {
        self.call(move |c| {
            Ok(c.query_row(
                "SELECT decision FROM confirmations WHERE job_id = ?1 ORDER BY created_at DESC, rowid DESC LIMIT 1",
                [job_id],
                |r| r.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten())
        })
        .await
    }

    // ---- audit ----

    pub async fn audit(
        &self,
        job_id: Option<i64>,
        delivery_id: Option<&str>,
        kind: &str,
        detail: Value,
    ) -> Result<()> {
        let delivery_id = delivery_id.map(str::to_string);
        let kind = kind.to_string();
        let detail = detail.to_string();
        self.call(move |c| {
            c.execute(
                "INSERT INTO audit (ts, job_id, delivery_id, kind, detail) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![now(), job_id, delivery_id, kind, detail],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn audit_entries(&self, job_id: i64) -> Result<Vec<(String, Value)>> {
        self.call(move |c| {
            let mut st =
                c.prepare("SELECT kind, detail FROM audit WHERE job_id = ?1 ORDER BY id")?;
            let rows = st
                .query_map([job_id], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows
                .into_iter()
                .map(|(k, d)| (k, serde_json::from_str(&d).unwrap_or(Value::String(d))))
                .collect())
        })
        .await
    }

    // ---- retention ----

    /// Audio files of deliveries received before `cutoff`; their paths are
    /// cleared in the same step so each file is reported once.
    pub async fn take_expired_audio(&self, cutoff: i64) -> Result<Vec<String>> {
        self.call(move |c| {
            let tx = c.transaction()?;
            let paths: Vec<String> = {
                let mut st = tx.prepare(
                    "SELECT audio_path FROM deliveries WHERE audio_path IS NOT NULL AND received_at < ?1",
                )?;
                st.query_map([cutoff], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            tx.execute(
                "UPDATE deliveries SET audio_path = NULL WHERE audio_path IS NOT NULL AND received_at < ?1",
                [cutoff],
            )?;
            tx.commit()?;
            Ok(paths)
        })
        .await
    }
}

/// Why a job could not be loaded.
enum LoadError {
    /// The row itself is bad: a column of the wrong type, invalid UTF-8, or
    /// an `AgentState` that no longer deserialises. Retrying cannot help.
    Unreadable(String),
    /// The database failed (I/O, busy...); worth another try.
    Db(anyhow::Error),
}

impl From<rusqlite::Error> for LoadError {
    fn from(e: rusqlite::Error) -> Self {
        use rusqlite::Error as E;
        match e {
            E::FromSqlConversionFailure(..)
            | E::IntegralValueOutOfRange(..)
            | E::Utf8Error(..)
            | E::InvalidColumnType(..) => LoadError::Unreadable(format!("unreadable job row: {e}")),
            e => LoadError::Db(e.into()),
        }
    }
}

fn load_job(c: &Connection, id: i64) -> Result<Option<Job>, LoadError> {
    let row = c
        .query_row(
            "SELECT j.id, j.delivery_id, j.profile, j.status, j.attempts, j.created_at, j.transcript, \
             j.transcript_source, j.agent_state, j.retry_notified, d.trigger, d.recorded_at_ms, \
             d.audio_path, d.phone_transcription \
             FROM jobs j JOIN deliveries d ON d.delivery_id = j.delivery_id WHERE j.id = ?1",
            [id],
            |r| {
                Ok((
                    Job {
                        id: r.get(0)?,
                        delivery_id: r.get(1)?,
                        profile: r.get(2)?,
                        status: JobStatus::parse(&r.get::<_, String>(3)?),
                        attempts: r.get(4)?,
                        created_at: r.get(5)?,
                        transcript: r.get(6)?,
                        transcript_source: r.get(7)?,
                        agent_state: None,
                        retry_notified: r.get(9)?,
                        trigger: r.get(10)?,
                        recorded_at_ms: r.get(11)?,
                        audio_path: r.get(12)?,
                        phone_transcription: r.get(13)?,
                    },
                    r.get::<_, Option<String>>(8)?,
                ))
            },
        )
        .optional()?;
    let Some((mut job, state)) = row else {
        return Ok(None);
    };
    if let Some(s) = state {
        job.agent_state = Some(
            serde_json::from_str(&s)
                .map_err(|e| LoadError::Unreadable(format!("decoding agent state: {e}")))?,
        );
    }
    Ok(Some(job))
}

/// Mark an unloadable job failed and audit it, salvaging whatever is still
/// readable so the user can be told what was lost.
fn quarantine(c: &Connection, id: i64, now: i64, reason: String) -> Result<Quarantined> {
    // Each column on its own: one bad column must not hide the others.
    let text = |r: &rusqlite::Row<'_>, i: usize| r.get::<_, Option<String>>(i).ok().flatten();
    let (delivery_id, transcript, phone) = c
        .query_row(
            "SELECT j.delivery_id, j.transcript, d.phone_transcription \
             FROM jobs j LEFT JOIN deliveries d ON d.delivery_id = j.delivery_id WHERE j.id = ?1",
            [id],
            |r| Ok((text(r, 0), text(r, 1), text(r, 2))),
        )
        .optional()?
        .unwrap_or_default();
    let error = format!("quarantined, not retried: {reason}");
    c.execute(
        "UPDATE jobs SET status = 'failed', last_error = ?2, updated_at = ?3 WHERE id = ?1",
        params![id, error, now],
    )?;
    c.execute(
        "INSERT INTO audit (ts, job_id, delivery_id, kind, detail) \
         VALUES (?1, ?2, ?3, 'job_quarantined', ?4)",
        params![
            now,
            id,
            delivery_id,
            serde_json::json!({ "error": error }).to_string()
        ],
    )?;
    Ok(Quarantined {
        id,
        delivery_id,
        heard: transcript.filter(|t| !t.trim().is_empty()).or(phone),
        error,
    })
}

pub fn now() -> i64 {
    jiff::Timestamp::now().as_second()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delivery(id: &str, profile: Option<&str>) -> NewDelivery {
        NewDelivery {
            delivery_id: id.into(),
            received_at: 1000,
            trigger: "double-click-hold".into(),
            is_test: false,
            recorded_at_ms: Some(999_000),
            client: Some("ring".into()),
            audio_path: Some("/data/audio/x.m4a".into()),
            audio_bytes: Some(10),
            phone_transcription: Some("hello".into()),
            profile: profile.map(Into::into),
        }
    }

    fn ready(claim: Option<Claim>) -> Job {
        match claim {
            Some(Claim::Ready(job)) => *job,
            other => panic!("expected a ready job, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unreadable_job_is_quarantined_not_left_at_the_head() {
        let db = Db::open_in_memory().unwrap();
        for id in ["bad", "good"] {
            db.insert_delivery(delivery(id, Some("command")), true)
                .await
                .unwrap();
        }
        db.call(|c| {
            c.execute(
                "UPDATE jobs SET agent_state = '{not json', transcript = 'turn on the hall light' \
                 WHERE delivery_id = 'bad'",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();

        let Some(Claim::Quarantined(q)) = db.claim_next_job(1000, &[]).await.unwrap() else {
            panic!("the bad job should be quarantined");
        };
        assert_eq!(q.delivery_id.as_deref(), Some("bad"));
        assert_eq!(q.heard.as_deref(), Some("turn on the hall light"));
        assert!(q.error.contains("decoding agent state"), "{}", q.error);
        // The good job behind it is next, and the bad one never comes back.
        let good = ready(db.claim_next_job(1000, &[]).await.unwrap());
        assert_eq!(good.delivery_id, "good");
        assert_eq!(db.requeue_running(1000, &[]).await.unwrap(), 1);
        assert_eq!(
            ready(db.claim_next_job(1000, &[]).await.unwrap()).delivery_id,
            "good"
        );
        assert!(db.claim_next_job(5000, &[]).await.unwrap().is_none());
        let kinds: Vec<String> = db
            .audit_entries(q.id)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(kinds, ["job_quarantined"]);
        // A column of the wrong type is quarantined too; the phone's
        // transcription is what is left of the request.
        db.insert_delivery(delivery("typed", Some("command")), true)
            .await
            .unwrap();
        db.call(|c| {
            c.execute(
                "UPDATE jobs SET created_at = 'yesterday' WHERE delivery_id = 'typed'",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
        let Some(Claim::Quarantined(q)) = db.claim_next_job(1000, &[]).await.unwrap() else {
            panic!("the mistyped job should be quarantined");
        };
        assert_eq!(q.heard.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn concurrent_claims_never_take_the_same_job() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.db");
        let db = Db::open(&path).unwrap();
        for i in 0..40 {
            db.insert_delivery(delivery(&format!("d{i}"), Some("command")), true)
                .await
                .unwrap();
        }
        // Separate connections, as two worker processes would have.
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let db = Db::open(&path).unwrap();
                tokio::spawn(async move {
                    let mut got = Vec::new();
                    while let Some(claim) = db.claim_next_job(1000, &[]).await.unwrap() {
                        got.push(ready(Some(claim)).id);
                    }
                    got
                })
            })
            .collect();
        let mut all = Vec::new();
        for w in workers {
            all.extend(w.await.unwrap());
        }
        all.sort_unstable();
        assert_eq!(all, (1..=40).collect::<Vec<i64>>());
    }

    #[tokio::test]
    async fn dedupe_and_claim() {
        let db = Db::open_in_memory().unwrap();
        let first = db
            .insert_delivery(delivery("a", Some("command")), true)
            .await
            .unwrap();
        assert!(matches!(first, InsertOutcome::Queued { .. }));
        let again = db
            .insert_delivery(delivery("a", Some("command")), true)
            .await
            .unwrap();
        assert_eq!(again, InsertOutcome::Duplicate);
        assert!(db.delivery_exists("a").await.unwrap());

        let job = ready(db.claim_next_job(1000, &[]).await.unwrap());
        assert_eq!(job.status, JobStatus::Running);
        assert_eq!(job.attempts, 1);
        assert_eq!(job.phone_transcription.as_deref(), Some("hello"));
        assert!(db.claim_next_job(1000, &[]).await.unwrap().is_none());

        // Jobs this process is still working on are left alone...
        assert_eq!(db.requeue_running(2000, &[job.id]).await.unwrap(), 0);
        // ...anything else left running (a restart) goes back.
        assert_eq!(db.requeue_running(2000, &[]).await.unwrap(), 1);
        // A job still in flight is not claimed twice.
        assert!(db.claim_next_job(2000, &[job.id]).await.unwrap().is_none());
        assert!(db.claim_next_job(2000, &[]).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn reschedule_respects_time() {
        let db = Db::open_in_memory().unwrap();
        db.insert_delivery(delivery("a", Some("command")), true)
            .await
            .unwrap();
        let job = ready(db.claim_next_job(1000, &[]).await.unwrap());
        db.reschedule(job.id, 1500, "mac asleep".into())
            .await
            .unwrap();
        assert!(db.claim_next_job(1499, &[]).await.unwrap().is_none());
        assert_eq!(db.next_due().await.unwrap(), Some(1500));
        assert!(db.claim_next_job(1500, &[]).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn confirmations_are_one_shot_and_expire() {
        let db = Db::open_in_memory().unwrap();
        let InsertOutcome::Queued { job_id } = db
            .insert_delivery(delivery("a", Some("command")), true)
            .await
            .unwrap()
        else {
            panic!()
        };
        db.set_status(job_id, JobStatus::AwaitingConfirmation)
            .await
            .unwrap();
        db.create_confirmation("h1".into(), job_id, "lock".into(), 2000)
            .await
            .unwrap();
        assert_eq!(
            db.decide("nope".into(), "approve".into(), 1500)
                .await
                .unwrap(),
            DecideOutcome::Unknown
        );
        assert_eq!(
            db.decide("h1".into(), "approve".into(), 1500)
                .await
                .unwrap(),
            DecideOutcome::Recorded { job_id }
        );
        assert_eq!(
            db.decide("h1".into(), "deny".into(), 1501).await.unwrap(),
            DecideOutcome::AlreadyDecided("approve".into())
        );
        assert_eq!(
            db.latest_decision(job_id).await.unwrap().as_deref(),
            Some("approve")
        );
        assert_eq!(
            db.get_job(job_id).await.unwrap().unwrap().status,
            JobStatus::Queued
        );

        db.set_status(job_id, JobStatus::AwaitingConfirmation)
            .await
            .unwrap();
        db.create_confirmation("h2".into(), job_id, "lock".into(), 3000)
            .await
            .unwrap();
        assert_eq!(
            db.decide("h2".into(), "approve".into(), 3000)
                .await
                .unwrap(),
            DecideOutcome::Expired
        );
        assert_eq!(db.expire_confirmations(3000).await.unwrap(), vec![job_id]);
        assert_eq!(
            db.latest_decision(job_id).await.unwrap().as_deref(),
            Some("expired")
        );
        assert_eq!(
            db.get_job(job_id).await.unwrap().unwrap().status,
            JobStatus::Queued
        );
    }

    #[tokio::test]
    async fn audio_retention() {
        let db = Db::open_in_memory().unwrap();
        db.insert_delivery(delivery("a", Some("command")), true)
            .await
            .unwrap();
        assert!(db.take_expired_audio(1000).await.unwrap().is_empty());
        assert_eq!(
            db.take_expired_audio(1001).await.unwrap(),
            vec!["/data/audio/x.m4a".to_string()]
        );
        assert!(db.take_expired_audio(5000).await.unwrap().is_empty());
    }
}
