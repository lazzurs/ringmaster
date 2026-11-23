//! The background worker: drains the job queue, retries with back-off while
//! the Mac or the API is unreachable, and does housekeeping.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentError, AgentOutcome, Request};
use crate::audio;
use crate::config::Config;
use crate::db::{Db, Job, JobStatus, now};
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
}

enum JobError {
    Transient(String),
    Permanent(String),
}

impl Worker {
    pub async fn run(self: Arc<Self>, shutdown: CancellationToken) {
        match self.db.requeue_running(now()).await {
            Ok(n) if n > 0 => tracing::info!(jobs = n, "requeued jobs interrupted by a restart"),
            Ok(_) => {}
            Err(e) => tracing::error!(error = %e, "requeue at start-up failed"),
        }
        let mut last_retention = None::<std::time::Instant>;
        loop {
            if shutdown.is_cancelled() {
                break;
            }
            self.tick(&shutdown).await;
            if last_retention.is_none_or(|t| t.elapsed() > Duration::from_secs(3600)) {
                self.retention().await;
                last_retention = Some(std::time::Instant::now());
            }
            let wait = match self.db.next_due().await {
                Ok(Some(due)) => Duration::from_secs((due - now()).clamp(1, 30) as u64),
                _ => Duration::from_secs(30),
            };
            tokio::select! {
                _ = self.wake.notified() => {}
                _ = tokio::time::sleep(wait) => {}
                _ = shutdown.cancelled() => break,
            }
        }
        tracing::info!("worker stopped");
    }

    /// Expire confirmations, then process every job that is due.
    pub async fn tick(&self, shutdown: &CancellationToken) {
        match self.db.expire_confirmations(now()).await {
            Ok(ids) => {
                for id in ids {
                    tracing::info!(job = id, "confirmation expired");
                }
            }
            Err(e) => tracing::error!(error = %e, "expiring confirmations failed"),
        }
        loop {
            if shutdown.is_cancelled() {
                break;
            }
            let job = match self.db.claim_next_job(now()).await {
                Ok(Some(job)) => job,
                Ok(None) => break,
                Err(e) => {
                    tracing::error!(error = %e, "claiming a job failed");
                    break;
                }
            };
            self.handle(job).await;
        }
        if let Ok(depth) = self.db.queue_depth().await {
            metrics::gauge!("ringmaster_queue_depth").set(depth as f64);
        }
    }

    #[tracing::instrument(skip_all, fields(job = job.id, delivery = %job.delivery_id, profile = %job.profile))]
    async fn handle(&self, job: Job) {
        let started = std::time::Instant::now();
        let result = self.process(&job).await;
        let outcome = match result {
            Ok(label) => label,
            Err(JobError::Transient(e)) => self.retry_later(&job, e).await,
            Err(JobError::Permanent(e)) => {
                tracing::warn!(error = %e, "job failed");
                self.fail(&job, &e).await;
                "failed"
            }
        };
        metrics::counter!("ringmaster_jobs_total", "outcome" => outcome).increment(1);
        metrics::histogram!("ringmaster_job_step_seconds").record(started.elapsed().as_secs_f64());
    }

    fn max_age(&self, job: &Job) -> i64 {
        self.config
            .profiles
            .get(&job.profile)
            .and_then(|p| p.job_max_age_secs)
            .unwrap_or(self.config.agent.job_max_age_secs) as i64
    }

    /// When the recording was made (or received, if earlier is unknown).
    fn origin(job: &Job) -> i64 {
        job.recorded_at_ms
            .map(|ms| ms / 1000)
            .map_or(job.created_at, |r| r.min(job.created_at))
    }

    async fn process(&self, job: &Job) -> Result<&'static str, JobError> {
        if job.agent_state.is_none() && now() - Self::origin(job) > self.max_age(job) {
            return Err(JobError::Permanent(
                "too old to run safely; not attempted".into(),
            ));
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
            Some(p) => {
                let path = self.data_dir.join(p);
                match tokio::fs::read(&path).await {
                    Ok(bytes) => {
                        match tokio::task::spawn_blocking(move || audio::m4a_to_whisper_wav(&bytes))
                            .await
                        {
                            Ok(Ok((wav, secs))) => {
                                tracing::debug!(secs, "decoded recording");
                                Some(wav)
                            }
                            Ok(Err(e)) => {
                                self.audit(
                                    job,
                                    "audio_decode_failed",
                                    json!({"error": format!("{e:#}")}),
                                )
                                .await;
                                None
                            }
                            Err(e) => {
                                return Err(JobError::Transient(format!("decoder task: {e}")));
                            }
                        }
                    }
                    Err(e) => {
                        self.audit(job, "audio_missing", json!({"error": e.to_string()}))
                            .await;
                        None
                    }
                }
            }
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
        let heard = job
            .transcript
            .as_deref()
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

fn short(s: &str) -> String {
    crate::agent::truncate(s.trim(), 160)
}
