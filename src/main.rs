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
    /// Compare models on synthetic voice commands, through the production
    /// request code. Tools come from fixtures; no MCP server is called.
    /// See docs/model-evaluation.md.
    Eval {
        /// The Ringmaster config: [llm] sections, keys from the environment,
        /// [approval] for the classifier suite.
        #[arg(
            long,
            env = "RINGMASTER_CONFIG",
            default_value = "/etc/ringmaster/config.toml"
        )]
        config: PathBuf,
        /// Fixture directory.
        #[arg(long, default_value = "eval/cases")]
        cases: PathBuf,
        /// Suites to run, comma-separated.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "agent,capture,judge,classifier"
        )]
        suite: Vec<String>,
        /// `provider:model` to evaluate (repeatable); default: the config's
        /// default model.
        #[arg(long = "model")]
        models: Vec<String>,
        /// Pad the tool list with this many irrelevant tools; several
        /// comma-separated counts run one pass each.
        #[arg(long, value_delimiter = ',', default_value = "0")]
        distractors: Vec<usize>,
        /// Run each case this many times.
        #[arg(long, default_value_t = 1)]
        repeat: u32,
        /// One JSON line per request.
        #[arg(long, default_value = "results.jsonl")]
        output: PathBuf,
        #[arg(long, default_value = "report.md")]
        report: PathBuf,
        /// Exit 2 if any harmful outcome occurred.
        #[arg(long)]
        fail_on_harmful: bool,
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

fn main() -> ExitCode {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("cannot start the runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let code = rt.block_on(run(Cli::parse()));
    // Dropping a runtime waits for every blocking thread. One that never
    // returns (a stuck decode or disk write) must not keep the process
    // alive until Docker's SIGKILL.
    rt.shutdown_timeout(Duration::from_secs(1));
    code
}

async fn run(cli: Cli) -> ExitCode {
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
                    // Docker keeps this output in `docker inspect`.
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    eprintln!(
                        "HTTP {status}: {}",
                        body.chars().take(500).collect::<String>()
                    );
                    ExitCode::FAILURE
                }
                Err(e) => {
                    eprintln!("unhealthy: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Eval {
            config,
            cases,
            suite,
            models,
            distractors,
            repeat,
            output,
            report,
            fail_on_harmful,
        } => {
            // Progress goes to stderr. Retry warnings name the endpoint, so
            // they are only shown with RUST_LOG set.
            let filter =
                EnvFilter::try_from_env("RUST_LOG").unwrap_or_else(|_| EnvFilter::new("error"));
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .init();
            let suites = match suite
                .iter()
                .map(|s| ringmaster::eval::Suite::parse(s))
                .collect::<anyhow::Result<Vec<_>>>()
            {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{e:#}");
                    return ExitCode::FAILURE;
                }
            };
            let opts = ringmaster::eval::Options {
                config,
                cases,
                suites,
                models,
                distractors,
                repeat,
                output,
                report,
            };
            match ringmaster::eval::run(&opts, &ringmaster::config::secret_from_env).await {
                Ok(s) => {
                    eprintln!(
                        "{} request(s); results in {}, report in {}",
                        s.rows,
                        opts.output.display(),
                        opts.report.display()
                    );
                    if s.harmful > 0 {
                        eprintln!("{} harmful outcome(s)", s.harmful);
                    }
                    if fail_on_harmful && s.harmful > 0 {
                        ExitCode::from(2)
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("eval failed: {e:#}");
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
                            ringmaster::config::display_url(&s.base_url),
                            s.api_key_env
                                .map(|v| format!(" (key from {v})"))
                                .unwrap_or_default()
                        );
                    }
                }
                if c.uses_classifier() {
                    let a = &c.approval;
                    let modes: Vec<String> = c
                        .profiles
                        .iter()
                        .map(|(name, p)| format!("{name}={}", c.approval_mode(Some(p)).as_str()))
                        .collect();
                    println!(
                        "  approval: {} at {} (auto-approve below {}, flagged above {}){}",
                        a.mode.as_str(),
                        ringmaster::config::display_url(a.base_url.as_deref().unwrap_or_default()),
                        a.auto_approve_cutoff(),
                        a.threshold,
                        if modes.is_empty() {
                            String::new()
                        } else {
                            format!("; profiles: {}", modes.join(", "))
                        }
                    );
                }
                let a = &c.auth_codes;
                if !a.codes.is_empty() {
                    let codes: Vec<String> = a
                        .codes
                        .iter()
                        .map(|(name, code)| format!("{name} (from {})", code.code_env))
                        .collect();
                    println!(
                        "  auth codes: {}; valid {} s; lockout after {} wrong in {} s, for {} s",
                        codes.join(", "),
                        a.valid_for_secs,
                        a.max_failures,
                        a.failure_window_secs,
                        a.lockout_secs
                    );
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
