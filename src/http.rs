//! HTTP routes: webhook, confirmations, health and metrics.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::sync::Notify;
use tower_http::trace::TraceLayer;

use crate::authcode::AuthCodes;
use crate::config::Config;
use crate::db::{Db, now};
use crate::health::Health;
use crate::notify::Notifier;
use crate::{confirm, webhook};

pub struct AppState {
    pub config: Arc<Config>,
    pub db: Db,
    pub notifier: Arc<dyn Notifier>,
    pub wake: Arc<Notify>,
    pub data_dir: PathBuf,
    pub webhook_secret: Option<String>,
    pub metrics: Option<PrometheusHandle>,
    pub health: Arc<Health>,
    /// Spoken authorisation codes, removed from the phone's transcription
    /// before it is stored.
    pub codes: Arc<AuthCodes>,
}

/// Longest `/healthz` waits for the database (Docker's probe allows 5 s).
const PING_LIMIT: std::time::Duration = std::time::Duration::from_secs(2);

pub fn router(state: Arc<AppState>) -> Router {
    let webhook_path = state.config.webhook.path.clone();
    let limit = state.config.server.max_body_bytes;
    Router::new()
        .route(
            &webhook_path,
            post(webhook::handle).layer(DefaultBodyLimit::max(limit)),
        )
        .route("/confirm/{token}", get(confirm::show))
        .route("/confirm/{token}/{decision}", post(confirm::decide))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .layer(TraceLayer::new_for_http().make_span_with(request_span))
        .with_state(state)
}

/// The request span logs the route template (`/confirm/{token}`), never the
/// path itself: a confirmation token in a debug log could approve a tool
/// call.
fn request_span(req: &Request) -> tracing::Span {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map_or("unmatched", MatchedPath::as_str);
    tracing::debug_span!("request", method = %req.method(), route, version = ?req.version())
}

/// `200 ok`, or `503` with the reasons, when recordings cannot be stored or
/// processed at all: the database is corrupt, read-only, full, failing I/O
/// or not answering, or the worker loop or the watchdog has stopped. It
/// reads in-memory state plus a bounded `SELECT 1`; it never runs the
/// expensive checks itself. Problems a restart cannot fix and that are not
/// the container's fault (failing jobs, a sleeping Mac, a down MCP server)
/// are alerted on but never make it unhealthy.
async fn healthz(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let now = now();
    let mut problems = app.health.unhealthy(now, &app.db.status().snapshot());
    match tokio::time::timeout(PING_LIMIT, app.db.ping()).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!(error = %e, "health check failed");
            problems.push("database unavailable".into());
        }
        Err(_) => problems.push("database unresponsive".into()),
    }
    if problems.is_empty() {
        (StatusCode::OK, "ok".to_string())
    } else {
        tracing::warn!(problems = ?problems, "unhealthy");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("unhealthy: {}", problems.join("; ")),
        )
    }
}

async fn metrics(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    match &app.metrics {
        Some(h) => {
            // Heartbeat ages and the like are as of this scrape.
            app.health.export(now(), &app.db.status().snapshot());
            (StatusCode::OK, h.render())
        }
        None => (StatusCode::NOT_FOUND, String::new()),
    }
}
