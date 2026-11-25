//! The background worker: drains the job queue, retries with back-off while
//! the Mac or the API is unreachable, and does housekeeping.
//!
//! One stuck job must never hold up the others. Each job runs in its own
//! task (up to `agent.worker_concurrency` at once), every attempt has a
//! wall-clock deadline (`agent.attempt_timeout_secs`), a panic fails only its
//! own job, a job that no longer loads is quarantined by the claim, and jobs
//! left `running` without a task (a restart, a lost task) are requeued.
//!
//! The loop beats a heartbeat in [`Health`] on every pass and reports the
//! jobs it runs and how they end, so the watchdog can tell a stalled loop,
//! a job whose deadline did not fire, or a run of failures.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::FutureExt;
use serde_json::json;
use tokio::sync::Notify;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentError, AgentOutcome, Request};
use crate::audio::{Decoded, Decoder};
use crate::config::{Config, WORKER_LOOP_MAX_WAIT_SECS};
use crate::db::{Claim, Db, Job, JobStatus, Quarantined, now};
use crate::health::Health;
use crate::notify::{Kind, Notification, Notifier};
use crate::stt::{SttError, Transcriber};

pub struct Worker {
    pub config: Arc<Config>,
    pub db: Db,
    pub transcriber: Transcriber,
    pub agent: Arc<Agent>,
    pub notifier: Arc<dyn Notifier>,
    pub wake: Arc<Notify>,
    pub data_dir: PathBuf,
    pub health: Arc<Health>,
    pub decoder: Decoder,
}

enum JobError {
    Transient(String),
    Permanent(String),
}

/// The jobs this worker is processing, one task each.
struct InFlight {
    tasks: JoinSet<()>,
    jobs: HashMap<tokio::task::Id, i64>,
    health: Arc<Health>,
}

impl InFlight {
    fn new(health: Arc<Health>) -> Self {
        Self {
            tasks: JoinSet::new(),
            jobs: HashMap::new(),
            health,
        }
    }

    fn len(&self) -> usize {
        self.tasks.len()
    }

    fn ids(&self) -> Vec<i64> {
        self.jobs.values().copied().collect()
    }

    fn spawn(&mut self, worker: Arc<Worker>, job: Job) {
        let id = job.id;
        self.health.job_started(id, now());
        let task = self.tasks.spawn(async move { worker.handle(job).await });
        self.jobs.insert(task.id(), id);
    }

    /// Wait for one job task to end; `None` when there are none.
    async fn join_next(&mut self) -> Option<()> {
        let res = self.tasks.join_next_with_id().await?;
        let task = match &res {
            Ok((id, ())) => *id,
            Err(e) => e.id(),
        };
        let job = self.jobs.remove(&task);
        if let Some(id) = job {
            self.health.job_finished(id);
        }
        if let Err(e) = res {
            // `handle` catches panics, so this is a panic while reporting
            // one. The job is still `running`; the next sweep requeues it.
            tracing::error!(job = ?job, error = %e, "job task ended abnormally");
        }
        Some(())
    }
}

impl Worker {
    pub async fn run(self: Arc<Self>, shutdown: CancellationToken) {
        let mut jobs = InFlight::new(self.health.clone());
        let mut last_retention = None::<std::time::Instant>;
        let max_wait = Duration::from_secs(WORKER_LOOP_MAX_WAIT_SECS);
        loop {
            if shutdown.is_cancelled() {
                break;
            }
            self.health.beat_worker(now());
            self.housekeeping(&jobs).await;
            self.start_due(&mut jobs, now(), &shutdown).await;
            self.record_depth().await;
            if last_retention.is_none_or(|t| t.elapsed() > Duration::from_secs(3600)) {
                self.retention().await;
                last_retention = Some(std::time::Instant::now());
            }
            // With every slot busy, a finishing job is what wakes the loop.
            let wait = match self.db.next_due().await {
                _ if jobs.len() >= self.config.agent.worker_concurrency => max_wait,
                Ok(Some(due)) => Duration::from_secs(
                    (due - now()).clamp(1, WORKER_LOOP_MAX_WAIT_SECS as i64) as u64,
                ),
                _ => max_wait,
            };
            tokio::select! {
                _ = self.wake.notified() => {}
                _ = tokio::time::sleep(wait) => {}
                Some(()) = jobs.join_next() => {}
                _ = shutdown.cancelled() => break,
            }
        }
        // Let the jobs in progress reach a save point; `run_service` stops
        // waiting after `server.shutdown_grace_secs` (dropping this task and
        // its job tasks) and anything unfinished is requeued at the next
        // start.
        while jobs.join_next().await.is_some() {}
        tracing::info!("worker stopped");
    }

    /// Process every job that is due now and wait for all of them (jobs that
    /// become due meanwhile, such as retries, wait for the next tick).
    pub async fn tick(self: &Arc<Self>, shutdown: &CancellationToken) {
        let due = now();
        let mut jobs = InFlight::new(self.health.clone());
        self.housekeeping(&jobs).await;
        loop {
            self.start_due(&mut jobs, due, shutdown).await;
            if jobs.join_next().await.is_none() {
                break;
            }
        }
        self.record_depth().await;
    }

    /// Expire confirmations and requeue jobs left `running` with no task
    /// working on them (a restart, a crash, a lost task).
    async fn housekeeping(&self, jobs: &InFlight) {
        match self.db.expire_confirmations(now()).await {
            Ok(ids) => {
                for id in ids {
                    tracing::info!(job = id, "confirmation expired");
                }
            }
            Err(e) => tracing::error!(error = %e, "expiring confirmations failed"),
        }
        match self.db.requeue_running(now(), &jobs.ids()).await {
            Ok(n) if n > 0 => tracing::info!(jobs = n, "requeued interrupted jobs"),
            Ok(_) => {}
            Err(e) => tracing::error!(error = %e, "requeueing interrupted jobs failed"),
        }
        // Close MCP sessions nobody has used for a while.
        self.agent.tools.housekeeping().await;
    }

    /// Start jobs that were due at `due` while there are free slots.
    async fn start_due(
        self: &Arc<Self>,
        jobs: &mut InFlight,
        due: i64,
        shutdown: &CancellationToken,
    ) {
        let slots = self.config.agent.worker_concurrency.max(1);
        while jobs.len() < slots && !shutdown.is_cancelled() {
            let claim = self.db.claim_next_job(due, &jobs.ids()).await;
            match &claim {
                Ok(_) => self.health.claim_ok(),
                Err(e) => self.health.claim_failed(crate::db::classify(e)),
            }
            match claim {
                Ok(Some(Claim::Ready(job))) => jobs.spawn(self.clone(), *job),
                Ok(Some(Claim::Quarantined(q))) => self.quarantined(q).await,
                Ok(None) => break,
                Err(e) => {
                    tracing::error!(error = %e, "claiming a job failed");
                    break;
                }
            }
        }
    }

    async fn record_depth(&self) {
        if let Ok(depth) = self.db.queue_depth().await {
            metrics::gauge!("ringmaster_queue_depth").set(depth as f64);
        }
    }

    /// Tell the user about a job that could not be loaded, with whatever is
    /// left of what they said, so the request is not silently lost.
    async fn quarantined(&self, q: Quarantined) {
        tracing::error!(job = q.id, delivery = ?q.delivery_id, error = %q.error, "quarantined a job that could not be loaded");
        metrics::counter!("ringmaster_jobs_total", "outcome" => "quarantined").increment(1);
        self.health.record_outcome(q.id, "quarantined", now());
        let said = q
            .heard
            .as_deref()
            .map(|t| format!(" You said: \"{}\"", crate::agent::truncate(t.trim(), 500)))
            .unwrap_or_default();
        self.notify(Notification {
            title: "Ringmaster: failed".into(),
            body: format!(
                "A queued request could not be read, so it was not run ({}).{said}",
                short(&q.error)
            ),
            kind: Kind::Error,
        })
        .await;
    }

    #[tracing::instrument(skip_all, fields(job = job.id, delivery = %job.delivery_id, profile = %job.profile))]
    async fn handle(&self, job: Job) {
        let started = std::time::Instant::now();
        // A panic fails this job only; the worker and the other jobs go on.
        let outcome = match AssertUnwindSafe(self.attempt(&job)).catch_unwind().await {
            Ok(outcome) => outcome,
            Err(panic) => {
                let msg = panic_message(panic.as_ref());
                tracing::error!(error = %msg, "job handling panicked");
                self.fail(&job, &format!("internal error: {msg}")).await;
                "panicked"
            }
        };
        metrics::counter!("ringmaster_jobs_total", "outcome" => outcome).increment(1);
        self.health.record_outcome(job.id, outcome, now());
        metrics::histogram!("ringmaster_job_step_seconds").record(started.elapsed().as_secs_f64());
    }

    /// One processing attempt, bounded by `agent.attempt_timeout_secs`. A
    /// hung attempt (a provider or MCP server that never answers) counts as
    /// a transient failure: back off and retry, until the job's max age.
    async fn attempt(&self, job: &Job) -> &'static str {
        let limit = Duration::from_secs(self.config.agent.attempt_timeout_secs);
        let result = tokio::time::timeout(limit, self.process(job))
            .await
            .unwrap_or_else(|_| {
                Err(JobError::Transient(format!(
                    "no result within {}s (agent.attempt_timeout_secs)",
                    limit.as_secs()
                )))
            });
        match result {
            Ok(label) => label,
            Err(JobError::Transient(e)) => self.retry_later(job, e).await,
            Err(JobError::Permanent(e)) => {
                tracing::warn!(error = %e, "job failed");
                self.fail(job, &e).await;
                "failed"
            }
        }
    }

    fn max_age(&self, job: &Job) -> i64 {
        self.config
            .profiles
            .get(&job.profile)
            .and_then(|p| p.job_max_age_secs)
            .unwrap_or(self.config.agent.job_max_age_secs) as i64
    }

    fn resume_grace(&self) -> i64 {
        let a = &self.config.agent;
        a.retry_max_secs.saturating_add(a.confirmation_ttl_secs) as i64
    }

    /// When the recording was made (or received, if earlier is unknown).
    fn origin(job: &Job) -> i64 {
        job.recorded_at_ms
            .map(|ms| ms / 1000)
            .map_or(job.created_at, |r| r.min(job.created_at))
    }

    async fn process(&self, job: &Job) -> Result<&'static str, JobError> {
        let age = now() - Self::origin(job);
        let max_age = self.max_age(job);
        match &job.agent_state {
            None if age > max_age => {
                return Err(JobError::Permanent(
                    "too old to run safely; not attempted".into(),
                ));
            }
            // Started jobs also come back after a restart or a lost task, not
            // only through `retry_later` (which stops at the max age). The
            // last scheduled retry, or the answer to a confirmation asked on
            // it, lands within `retry_max_secs + confirmation_ttl_secs` of
            // the max age; past that (a job that keeps crashing the service,
            // a long outage) a half-done request is stopped, not resumed.
            // An answer that was already given is still delivered.
            Some(state) if state.final_text.is_none() && age > max_age + self.resume_grace() => {
                return Err(JobError::Permanent(format!(
                    "too old to resume safely; stopped after {} attempts",
                    job.attempts
                )));
            }
            _ => {}
        }

        let (text, source) = match (&job.transcript, &job.transcript_source) {
            (Some(t), Some(s)) => (t.clone(), s.clone()),
            _ => self.transcribe(job).await?,
        };

        if text.trim().is_empty() {
            self.finish(
                job,
                JobStatus::Done,
                "I didn't catch anything in that recording.",
                Kind::Info,
            )
            .await;
            return Ok("empty");
        }

        let req = Request {
            transcript: &text,
            transcript_source: &source,
        };
        match self.agent.run(job, &req).await {
            Ok(AgentOutcome::Final(reply)) => {
                self.finish(job, JobStatus::Done, &reply, Kind::Result)
                    .await;
                Ok("done")
            }
            Ok(AgentOutcome::AwaitingConfirmation) => Ok("awaiting_confirmation"),
            Ok(AgentOutcome::Refused(msg)) => {
                self.finish(job, JobStatus::Failed, &msg, Kind::Error).await;
                Ok("refused")
            }
            Ok(AgentOutcome::GaveUp(msg)) => {
                self.finish(job, JobStatus::Failed, &msg, Kind::Error).await;
                Ok("gave_up")
            }
            Err(AgentError::Transient(e)) => Err(JobError::Transient(e)),
            Err(AgentError::Permanent(e)) => Err(JobError::Permanent(e)),
        }
    }

    async fn transcribe(&self, job: &Job) -> Result<(String, String), JobError> {
        let wav = match &job.audio_path {
            None => None,
            Some(p) => self.decode(job, &self.data_dir.join(p)).await?,
        };
        match self
            .transcriber
            .transcribe(wav.as_deref(), job.phone_transcription.as_deref())
            .await
        {
            Ok(t) => {
                self.audit(
                    job,
                    "transcript",
                    json!({"source": t.source, "text": t.text}),
                )
                .await;
                self.db
                    .save_transcript(job.id, t.text.clone(), t.source.to_string())
                    .await
                    .map_err(|e| JobError::Transient(e.to_string()))?;
                Ok((t.text, t.source.to_string()))
            }
            Err(SttError::Transient(e)) => Err(JobError::Transient(e)),
            Err(SttError::Permanent(e)) => Err(JobError::Permanent(e)),
        }
    }

    /// The recording as a WAV, or `None` to go on with the phone's
    /// transcription (missing, oversized or undecodable file, a decoder
    /// that ran out of time or panicked). With every decoder still stuck on
    /// earlier recordings and no phone transcription to use, the job waits
    /// for a later attempt instead.
    async fn decode(&self, job: &Job, path: &std::path::Path) -> Result<Option<Vec<u8>>, JobError> {
        let too_big = match tokio::fs::metadata(path).await {
            Ok(m) => m.len() > self.decoder.max_bytes(),
            Err(e) => {
                self.audit(job, "audio_missing", json!({"error": e.to_string()}))
                    .await;
                return Ok(None);
            }
        };
        let decoded = if too_big {
            Decoded::Failed(format!(
                "recording is over stt.max_audio_bytes ({})",
                self.decoder.max_bytes()
            ))
        } else {
            match tokio::fs::read(path).await {
                Ok(bytes) => self.decoder.decode(bytes).await,
                Err(e) => {
                    self.audit(job, "audio_missing", json!({"error": e.to_string()}))
                        .await;
                    return Ok(None);
                }
            }
        };
        match decoded {
            Decoded::Wav(wav, secs) => {
                tracing::debug!(secs, "decoded recording");
                Ok(Some(wav))
            }
            // As for any undecodable recording (a panic included), fall
            // back to the phone's transcription rather than retrying.
            Decoded::Failed(error) => {
                tracing::warn!(error = %error, "could not decode recording");
                self.audit(job, "audio_decode_failed", json!({"error": error}))
                    .await;
                Ok(None)
            }
            Decoded::Busy => {
                let error = "every audio decoder is busy (stuck on an earlier recording?)";
                tracing::warn!(error, "recording not decoded");
                self.audit(job, "audio_decode_failed", json!({"error": error}))
                    .await;
                let usable = self.config.stt.use_phone_transcription
                    && job
                        .phone_transcription
                        .as_deref()
                        .is_some_and(|t| !t.trim().is_empty());
                if usable {
                    Ok(None)
                } else {
                    Err(JobError::Transient(error.into()))
                }
            }
        }
    }

    async fn retry_later(&self, job: &Job, error: String) -> &'static str {
        let age = now() - Self::origin(job);
        let max_age = self.max_age(job);
        if age >= max_age {
            let msg = format!("gave up after {} attempts: {error}", job.attempts);
            tracing::warn!(error = %msg, "job expired");
            self.fail(job, &msg).await;
            return "expired";
        }
        let cfg = &self.config.agent;
        let exp = (job.attempts.max(1) - 1).min(16) as u32;
        let delay = cfg
            .retry_base_secs
            .saturating_mul(2u64.saturating_pow(exp))
            .min(cfg.retry_max_secs)
            .max(1) as i64;
        let next = now() + delay;
        tracing::info!(error = %error, delay, "job delayed; will retry");
        self.audit(
            job,
            "retry_scheduled",
            json!({"error": error, "delay_secs": delay}),
        )
        .await;
        if let Err(e) = self.db.reschedule(job.id, next, error.clone()).await {
            tracing::error!(error = %e, "rescheduling failed");
        }
        if cfg.notify_on_retry && !job.retry_notified {
            let remaining = (max_age - age).max(0) / 60;
            self.notify(Notification {
                title: "Ringmaster: delayed".into(),
                body: format!("Couldn't process that yet ({}). I'll keep trying for about {remaining} more minutes.", short(&error)),
                kind: Kind::Error,
            })
            .await;
            let _ = self.db.mark_retry_notified(job.id).await;
        }
        "retry"
    }

    async fn fail(&self, job: &Job, error: &str) {
        let _ = self
            .db
            .finish_job(job.id, JobStatus::Failed, None, Some(error.to_string()))
            .await;
        self.audit(job, "job_failed", json!({"error": error})).await;
        // `job` is the row as claimed; the transcript may have been saved
        // since, in this attempt.
        let saved = match &job.transcript {
            Some(_) => None,
            None => self.db.get_job(job.id).await.ok().flatten(),
        };
        let heard = job
            .transcript
            .as_deref()
            .or_else(|| saved.as_ref().and_then(|j| j.transcript.as_deref()))
            .or(job.phone_transcription.as_deref())
            .map(|t| format!(" (\"{}\")", short(t)))
            .unwrap_or_default();
        self.notify(Notification {
            title: "Ringmaster: failed".into(),
            body: format!("That request{heard} failed: {}", short(error)),
            kind: Kind::Error,
        })
        .await;
    }

    async fn finish(&self, job: &Job, status: JobStatus, text: &str, kind: Kind) {
        if let Err(e) = self
            .db
            .finish_job(job.id, status, Some(text.to_string()), None)
            .await
        {
            tracing::error!(error = %e, "storing the job result failed");
        }
        self.audit(
            job,
            "job_finished",
            json!({"status": status.as_str(), "text": text}),
        )
        .await;
        let title = match kind {
            Kind::Error => "Ringmaster: problem",
            _ => "Ringmaster",
        };
        self.notify(Notification {
            title: title.into(),
            body: text.to_string(),
            kind,
        })
        .await;
    }

    async fn notify(&self, n: Notification) {
        if let Err(e) = self.notifier.send(&n).await {
            tracing::error!(error = %e, "notification failed");
        }
    }

    async fn audit(&self, job: &Job, kind: &str, detail: serde_json::Value) {
        if let Err(e) = self
            .db
            .audit(Some(job.id), Some(&job.delivery_id), kind, detail)
            .await
        {
            tracing::error!(error = %e, "audit write failed");
        }
    }

    /// Delete recordings past the retention period.
    pub async fn retention(&self) {
        let days = self.config.retention.audio_days;
        if days == 0 {
            return;
        }
        let cutoff = now() - i64::from(days) * 86_400;
        match self.db.take_expired_audio(cutoff).await {
            Ok(paths) => {
                for p in paths {
                    let full = self.data_dir.join(&p);
                    match tokio::fs::remove_file(&full).await {
                        Ok(()) => tracing::info!(path = %p, "deleted expired recording"),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => {
                            tracing::warn!(path = %p, error = %e, "could not delete recording")
                        }
                    }
                }
            }
            Err(e) => tracing::error!(error = %e, "retention sweep failed"),
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

fn short(s: &str) -> String {
    crate::agent::truncate(s.trim(), 160)
}
