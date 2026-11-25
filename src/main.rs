use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "ringmaster", version = ringmaster::VERSION, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the service (default).
    Serve {
        #[arg(
            long,
            env = "RINGMASTER_CONFIG",
            default_value = "/etc/ringmaster/config.toml"
        )]
        config: PathBuf,
    },
    /// Probe /healthz and exit 0 if healthy (for Docker HEALTHCHECK; the
    /// runtime image has no curl).
    Healthcheck {
        #[arg(
            long,
            env = "RINGMASTER_HEALTHCHECK_URL",
            default_value = "http://127.0.0.1:8080/healthz"
        )]
        url: String,
    },
    /// Validate a config file and exit.
    CheckConfig {
        #[arg(
            long,
            env = "RINGMASTER_CONFIG",
            default_value = "/etc/ringmaster/config.toml"
        )]
        config: PathBuf,
    },
}

fn init_logging() {
    let filter = EnvFilter::try_from_env("RUST_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info,ringmaster=info,tower_http=info"));
    let text = std::env::var("RINGMASTER_LOG_FORMAT").is_ok_and(|v| v == "text");
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    if text {
        builder.init();
    } else {
        builder
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(false)
            .init();
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Command::Serve {
        config: std::env::var_os("RINGMASTER_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/etc/ringmaster/config.toml")),
    });
    match command {
        Command::Serve { config } => {
            init_logging();
            match ringmaster::serve(&config).await {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    tracing::error!(error = format!("{e:#}"), "fatal");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Healthcheck { url } => {
            let client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .no_proxy()
                .build()
            {
                Ok(c) => c,
                Err(_) => return ExitCode::FAILURE,
            };
            match client.get(&url).send().await {
                Ok(r) if r.status().is_success() => ExitCode::SUCCESS,
                Ok(r) => {
                    eprintln!("unhealthy: HTTP {}", r.status());
                    ExitCode::FAILURE
                }
                Err(e) => {
                    eprintln!("unhealthy: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::CheckConfig { config } => match ringmaster::config::Config::load(&config) {
            Ok(c) if !c.mcp_warnings.is_empty() => {
                // `serve` would start without these servers; flag them here.
                for w in &c.mcp_warnings {
                    eprintln!("warning: {w}");
                }
                ExitCode::FAILURE
            }
            Ok(c) => {
                println!(
                    "ok: {} profile(s), {} MCP server(s), {} tool policy rule(s)",
                    c.profiles.len(),
                    c.mcp_servers.len(),
                    c.tool_policies.len()
                );
                for (name, p) in &c.profiles {
                    // Validated by load(), so this resolves.
                    if let Ok(choice) = c.profile_llm(Some(p)) {
                        let s = c.provider_settings(choice.provider);
                        println!(
                            "  profile {name}: {} {} at {}{}",
                            choice.provider,
                            choice.model,
                            s.base_url,
                            s.api_key_env
                                .map(|v| format!(" (key from {v})"))
                                .unwrap_or_default()
                        );
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("invalid config: {e:#}");
                ExitCode::FAILURE
            }
        },
    }
}
