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

use crate::config::Config;
use crate::db::Db;
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
}

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

async fn healthz(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    match app.db.ping().await {
        Ok(()) => (StatusCode::OK, "ok"),
        Err(e) => {
            tracing::error!(error = %e, "health check failed");
            (StatusCode::SERVICE_UNAVAILABLE, "database unavailable")
        }
    }
}

async fn metrics(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    match &app.metrics {
        Some(h) => (StatusCode::OK, h.render()),
        None => (StatusCode::NOT_FOUND, String::new()),
    }
}
