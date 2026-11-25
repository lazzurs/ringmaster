//! Ringmaster: voice commands from a smart ring.
//!
//! One tokio process: the webhook receiver persists recordings and jobs in
//! SQLite and answers at once; the worker transcribes (whisper.cpp on a Mac),
//! runs a Claude agent over MCP tools, and reports back through ntfy.

pub mod agent;
pub mod audio;
pub mod claude;
pub mod config;
pub mod confirm;
pub mod db;
pub mod http;
pub mod notify;
pub mod policy;
pub mod signature;
pub mod stt;
pub mod tools;
pub mod webhook;
pub mod worker;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::claude::ClaudeClient;
use crate::config::{Config, Secrets};
use crate::db::Db;
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

    let data_dir = config.server.data_dir.clone();
    std::fs::create_dir_all(data_dir.join("audio"))
        .with_context(|| format!("creating {}", data_dir.display()))?;
    let db = Db::open(&data_dir.join("ringmaster.db"))?;

    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| tracing::warn!(error = %e, "metrics recorder not installed"))
        .ok();

    let notifier: Arc<dyn Notifier> = match &config.notify.ntfy {
        Some(n) => Arc::new(NtfyNotifier::new(n, secrets.ntfy_token.clone())?),
        None => Arc::new(LogNotifier),
    };
    let claude = ClaudeClient::new(
        &config.claude.api_base,
        secrets.anthropic_api_key.clone().unwrap_or_default(),
        Duration::from_secs(config.claude.request_timeout_secs),
        config.claude.max_retries,
    )?;
    let tools: Arc<dyn ToolBackend> = Arc::new(McpBackend::new(
        config.mcp_servers.clone(),
        secrets.mcp_headers.clone(),
        Duration::from_secs(config.agent.tool_timeout_secs),
    ));
    let agent = Arc::new(Agent::new(
        config.clone(),
        db.clone(),
        claude,
        tools,
        notifier.clone(),
    )?);
    let wake = Arc::new(Notify::new());
    let worker = Arc::new(Worker {
        config: config.clone(),
        db: db.clone(),
        transcriber: Transcriber::new(config.stt.clone())?,
        agent,
        notifier: notifier.clone(),
        wake: wake.clone(),
        data_dir: data_dir.clone(),
    });

    let state = Arc::new(http::AppState {
        config: config.clone(),
        db: db.clone(),
        notifier,
        wake,
        data_dir,
        webhook_secret: secrets.webhook_secret.clone(),
        metrics,
    });

    let shutdown = CancellationToken::new();
    let worker_task = tokio::spawn(worker.run(shutdown.child_token()));

    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .with_context(|| format!("binding {}", config.server.listen))?;
    tracing::info!(addr = %config.server.listen, "listening");
    let signal_token = shutdown.clone();
    axum::serve(listener, http::router(state))
        .with_graceful_shutdown(async move {
            wait_for_signal().await;
            tracing::info!("shutting down");
            signal_token.cancel();
        })
        .await?;

    // Let the job in progress reach a save point; anything unfinished is
    // requeued at the next start.
    shutdown.cancel();
    if tokio::time::timeout(Duration::from_secs(20), worker_task)
        .await
        .is_err()
    {
        tracing::warn!("worker did not stop in time; its job will resume after restart");
    }
    Ok(())
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
