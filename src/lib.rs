//! Ringmaster: voice commands from a smart ring.
//!
//! One tokio process: the webhook receiver persists recordings and jobs in
//! SQLite and answers at once; the worker transcribes (whisper.cpp on a Mac),
//! runs an LLM agent (Claude, or another provider) over MCP tools, and
//! reports back through ntfy.

pub mod agent;
pub mod audio;
pub mod classifier;
pub mod claude;
pub mod config;
pub mod confirm;
pub mod db;
pub mod health;
pub mod http;
pub mod llm;
pub mod notify;
pub mod policy;
pub mod signature;
pub mod stt;
pub mod tools;
pub mod webhook;
pub mod worker;

use std::future::IntoFuture;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::audio::Decoder;
use crate::config::{Config, Secrets};
use crate::db::Db;
use crate::health::{AlertBook, Alerter, Health, Watchdog};
use crate::llm::Llms;
use crate::notify::{LogNotifier, Notifier, NtfyNotifier};
use crate::stt::Transcriber;
use crate::tools::{McpBackend, ToolBackend};
use crate::worker::Worker;

/// Version string: the release tag when built by CI, else Cargo's.
pub const VERSION: &str = match option_env!("RINGMASTER_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

pub async fn serve(config_path: &Path) -> Result<()> {
    let config = Arc::new(Config::load(config_path)?);
    let secrets = Secrets::from_env(&config)?;
    config.validate_runtime(&secrets)?;
    tracing::info!(version = VERSION, config = %config_path.display(), secrets = ?secrets, "starting");
    for w in config.mcp_warnings.iter().chain(&secrets.warnings) {
        tracing::warn!("{w}");
    }

    // Signals are watched from the start, so SIGTERM also ends the wait for
    // a broken database below.
    let shutdown = CancellationToken::new();
    let on_signal = shutdown.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = wait_for_signal() => {
                tracing::info!("shutting down");
                on_signal.cancel();
            }
            _ = on_signal.cancelled() => {}
        }
    });

    let data_dir = config.server.data_dir.clone();
    std::fs::create_dir_all(data_dir.join("audio"))
        .with_context(|| format!("creating {}", data_dir.display()))?;

    let notifier: Arc<dyn Notifier> = match &config.notify.ntfy {
        Some(n) => Arc::new(NtfyNotifier::new(n, secrets.ntfy_token.clone())?),
        None => Arc::new(LogNotifier),
    };
    let alert_notifier: Arc<dyn Notifier> = match (&config.notify.ntfy, &config.alerts.ntfy_topic) {
        (Some(n), Some(topic)) => Arc::new(NtfyNotifier::with_topic(
            n,
            topic,
            secrets.ntfy_token.clone(),
        )?),
        _ => notifier.clone(),
    };
    let alerter = Alerter::new(
        alert_notifier,
        &config.alerts,
        AlertBook::load(Some(data_dir.join("alerts.json"))),
    );

    let Some(db) = open_db(&data_dir.join("ringmaster.db"), &alerter, &shutdown).await else {
        return Ok(());
    };

    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| tracing::warn!(error = %e, "metrics recorder not installed"))
        .ok();

    let health = Arc::new(Health::new(&config, db::now()));
    let llms = Llms::new(&config, &secrets.llm_keys)?;
    let tools: Arc<dyn ToolBackend> = Arc::new(McpBackend::new(
        config.mcp_servers.clone(),
        secrets.mcp_headers.clone(),
        Duration::from_secs(config.agent.tool_timeout_secs),
    ));
    let classifier = classifier::Classifier::from_config(&config, secrets.classifier_key.clone())?
        .map(|c| Arc::new(c.with_health(health.clone())));
    if let Some(base) = &config.approval.base_url
        && classifier.is_some()
    {
        tracing::info!(
            mode = config.approval.mode.as_str(),
            url = %config::display_url(base),
            "approval classifier configured"
        );
    }
    let agent = Arc::new(
        Agent::new(config.clone(), db.clone(), llms, tools, notifier.clone())?
            .with_classifier(classifier),
    );
    let wake = Arc::new(Notify::new());
    let worker = Arc::new(Worker {
        config: config.clone(),
        db: db.clone(),
        transcriber: Transcriber::new(config.stt.clone())?,
        agent,
        notifier: notifier.clone(),
        wake: wake.clone(),
        data_dir: data_dir.clone(),
        health: health.clone(),
        decoder: Decoder::new(&config.stt, health.blocking.clone()),
    });
    let watchdog =
        Arc::new(Watchdog::new(health.clone(), db.clone(), alerter).with_metrics(metrics.clone()));

    let state = Arc::new(http::AppState {
        config: config.clone(),
        db: db.clone(),
        notifier,
        wake,
        data_dir,
        webhook_secret: secrets.webhook_secret.clone(),
        metrics,
        health,
    });

    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .with_context(|| format!("binding {}", config.server.listen))?;
    tracing::info!(addr = %config.server.listen, "listening");
    let report = run_service(
        listener,
        http::router(state),
        worker,
        watchdog,
        shutdown,
        Duration::from_secs(config.server.shutdown_grace_secs),
    )
    .await?;
    if !report.jobs_left.is_empty() {
        tracing::warn!(jobs = ?report.jobs_left, "jobs still running at shutdown; they resume after restart");
    }
    Ok(())
}

/// Open the database, retrying every 30 s while it cannot be opened (a
/// corrupt file, a full or read-only disk) instead of exiting: exiting would
/// make the container restart in a loop. Alerts while it waits. `None` when
/// a shutdown was requested meanwhile.
async fn open_db(path: &Path, alerter: &Alerter, shutdown: &CancellationToken) -> Option<Db> {
    use crate::health::Condition;
    loop {
        match Db::open(path) {
            Ok(db) => return Some(db),
            Err(e) => {
                let class = db::classify(&e);
                tracing::error!(
                    error = format!("{e:#}"),
                    "cannot open the database; retrying in 30 s"
                );
                let what = class.map_or("an unknown error".to_string(), |c| c.to_string());
                let condition = if class.is_some_and(|c| c.is_corruption()) {
                    Condition::DatabaseCorrupt
                } else {
                    Condition::DatabaseFailing
                };
                let detail = format!(
                    "Ringmaster cannot open its database ({what}) and is not accepting \
                     recordings; it retries every 30 s. New recordings are not stored or \
                     processed while this lasts."
                );
                alerter
                    .apply(db::now(), &[(condition, detail)].into())
                    .await;
            }
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(30)) => {}
            _ = shutdown.cancelled() => return None,
        }
    }
}

/// How a shutdown went.
#[derive(Debug, Default)]
pub struct ShutdownReport {
    /// Jobs still running when the grace period ran out. Their rows stay
    /// `running` and the next start requeues them.
    pub jobs_left: Vec<i64>,
    /// HTTP requests still open when the grace period ran out were dropped.
    pub http_cut_off: bool,
}

/// Run the HTTP server, the worker and the watchdog until `shutdown` is
/// cancelled (a signal), then stop: the listener closes at once (new
/// webhooks are refused, so the app retries elsewhere or later), in-flight
/// requests and jobs get `grace` to finish, the watchdog and housekeeping
/// stop, and whatever is left is dropped. Every task this starts has ended
/// when it returns.
pub async fn run_service(
    listener: tokio::net::TcpListener,
    router: axum::Router,
    worker: Arc<Worker>,
    watchdog: Arc<Watchdog>,
    shutdown: CancellationToken,
    grace: Duration,
) -> Result<ShutdownReport> {
    let health = worker.health.clone();
    let mut worker_task = tokio::spawn(worker.run(shutdown.child_token()));
    let mut watchdog_task = tokio::spawn(watchdog.run(shutdown.child_token()));
    let stop = shutdown.clone();
    let mut server = tokio::spawn(
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { stop.cancelled().await })
            .into_future(),
    );

    let failed = tokio::select! {
        _ = shutdown.cancelled() => None,
        r = &mut server => Some(r),
    };
    shutdown.cancel();
    let deadline = tokio::time::Instant::now() + grace;
    let mut report = ShutdownReport::default();

    let served = match failed {
        Some(r) => Some(r),
        None => match tokio::time::timeout_at(deadline, &mut server).await {
            Ok(r) => Some(r),
            Err(_) => {
                server.abort();
                report.http_cut_off = true;
                None
            }
        },
    };
    if tokio::time::timeout_at(deadline, &mut worker_task)
        .await
        .is_err()
    {
        report.jobs_left = health.running_jobs();
        // Dropping the worker's task set cancels its job tasks.
        worker_task.abort();
        let _ = worker_task.await;
    }
    if tokio::time::timeout_at(deadline + Duration::from_secs(1), &mut watchdog_task)
        .await
        .is_err()
    {
        watchdog_task.abort();
    }
    match served {
        Some(Ok(Ok(()))) | None => Ok(report),
        Some(Ok(Err(e))) => Err(e).context("HTTP server failed"),
        Some(Err(e)) => Err(e).context("HTTP server task failed"),
    }
}
async fn wait_for_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("installing SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
}
