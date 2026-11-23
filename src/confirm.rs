//! Approve/Deny endpoints behind the notification buttons.
//!
//! `POST /confirm/{token}/{approve|deny}` is what the ntfy `http` actions call.
//! `GET /confirm/{token}` shows the pending action with two buttons, for when
//! the notification is opened in a browser. GET never changes anything, so
//! link previews and prefetching cannot approve an action.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use serde_json::json;

use crate::agent::hash_token;
use crate::db::{DecideOutcome, now};
use crate::http::AppState;

fn valid_token(t: &str) -> bool {
    t.len() == 43
        && t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn page(status: StatusCode, title: &str, body: &str) -> Response {
    let html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta name=\"robots\" content=\"noindex\"><title>{t}</title>\
         <style>body{{font-family:system-ui,sans-serif;max-width:36rem;margin:2rem auto;padding:0 1rem;line-height:1.5}}\
         pre{{white-space:pre-wrap;background:#f2f2f2;padding:.75rem;border-radius:6px}}\
         button{{font-size:1.1rem;padding:.6rem 1.4rem;margin-right:.75rem;border-radius:6px;border:1px solid #888}}\
         @media (prefers-color-scheme: dark){{body{{background:#111;color:#eee}}pre{{background:#222}}}}</style>\
         </head><body><h1>{t}</h1>{body}</body></html>",
        t = escape(title)
    );
    (status, Html(html)).into_response()
}

pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub async fn show(State(app): State<Arc<AppState>>, Path(token): Path<String>) -> Response {
    if !valid_token(&token) {
        return page(
            StatusCode::NOT_FOUND,
            "Not found",
            "<p>Unknown or expired link.</p>",
        );
    }
    match app.db.get_confirmation(hash_token(&token)).await {
        Ok(Some(c)) => {
            let summary = format!("<pre>{}</pre>", escape(&c.summary));
            if let Some(d) = c.decision {
                return page(
                    StatusCode::GONE,
                    "Already answered",
                    &format!("{summary}<p>Decision: <strong>{}</strong></p>", escape(&d)),
                );
            }
            if c.expires_at <= now() {
                return page(
                    StatusCode::GONE,
                    "Expired",
                    &format!("{summary}<p>This request has expired.</p>"),
                );
            }
            page(
                StatusCode::OK,
                "Approve this action?",
                &format!(
                    "{summary}<form method=\"post\" action=\"{t}/approve\" style=\"display:inline\">\
                     <button type=\"submit\">Approve</button></form>\
                     <form method=\"post\" action=\"{t}/deny\" style=\"display:inline\">\
                     <button type=\"submit\">Deny</button></form>",
                    t = escape(&token)
                ),
            )
        }
        Ok(None) => page(
            StatusCode::NOT_FOUND,
            "Not found",
            "<p>Unknown or expired link.</p>",
        ),
        Err(e) => {
            tracing::error!(error = %e, "database error");
            page(
                StatusCode::SERVICE_UNAVAILABLE,
                "Unavailable",
                "<p>Try again shortly.</p>",
            )
        }
    }
}

pub async fn decide(
    State(app): State<Arc<AppState>>,
    Path((token, decision)): Path<(String, String)>,
) -> Response {
    let decision = match decision.as_str() {
        "approve" => "approve",
        "deny" => "deny",
        _ => return (StatusCode::NOT_FOUND, "unknown decision").into_response(),
    };
    if !valid_token(&token) {
        return (StatusCode::NOT_FOUND, "unknown or expired").into_response();
    }
    match app
        .db
        .decide(hash_token(&token), decision.to_string(), now())
        .await
    {
        Ok(DecideOutcome::Recorded { job_id }) => {
            metrics::counter!("ringmaster_confirmations_total", "decision" => decision)
                .increment(1);
            let _ = app
                .db
                .audit(
                    Some(job_id),
                    None,
                    "confirmation_answered",
                    json!({"decision": decision}),
                )
                .await;
            tracing::info!(job = job_id, decision, "confirmation answered");
            app.wake.notify_one();
            let msg = if decision == "approve" {
                "Approved."
            } else {
                "Denied."
            };
            page(StatusCode::OK, msg, "<p>You can close this page.</p>")
        }
        Ok(DecideOutcome::AlreadyDecided(d)) => page(
            StatusCode::GONE,
            "Already answered",
            &format!("<p>Decision: <strong>{}</strong></p>", escape(&d)),
        ),
        Ok(DecideOutcome::Expired) => page(
            StatusCode::GONE,
            "Expired",
            "<p>This request has expired.</p>",
        ),
        Ok(DecideOutcome::Unknown) => (StatusCode::NOT_FOUND, "unknown or expired").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "database error");
            (StatusCode::SERVICE_UNAVAILABLE, "try again").into_response()
        }
    }
}
