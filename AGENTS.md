# AGENTS.md

Guide for coding agents working on Ringmaster. Read this first; the README is
the user-facing reference.

## What Ringmaster is

One Rust service (one tokio process, one container) that turns voice commands
from a smart ring into actions:

```
POST /webhook (HMAC-signed multipart) -> SQLite job queue -> worker
  -> whisper.cpp /inference (primary, fallback, then the phone's transcript)
  -> LLM agent loop (Anthropic native; OpenAI/Mistral/OpenRouter chat completions; Ollama /api/chat)
  -> MCP tools over streamable HTTP, gated by allow/confirm/ask/deny/code policies
     (on the tool name and optionally its arguments; `code` needs a spoken
     authorisation code, checked by Ringmaster after transcription;
     optionally: a local Shieldstral classifier approves clear `confirm` passes)
  -> ntfy notification (result, problem, or Approve/Deny buttons)
```

Jobs are durable: the conversation is saved after every step, so restarts,
outages and pending confirmations resume where they stopped.

## Code map

| Path | What lives there | Change it for |
|---|---|---|
| `src/main.rs` | CLI: `serve`, `healthcheck`, `check-config`, `eval`; logging setup; the runtime (built by hand so `shutdown_timeout` drops stuck blocking threads) | new subcommands, `check-config` output |
| `src/lib.rs` | `serve()`: wires config, secrets, DB (`open_db` retries and alerts instead of exiting), notifier and alert notifier, LLMs, MCP backend, worker, watchdog, HTTP; `run_service()`: runs them and shuts down within `server.shutdown_grace_secs` | start-up wiring, shutdown |
| `src/health.rs` | `Health` (heartbeats, jobs in flight, failures, `BlockingTracker`, gauges, `/healthz` reasons), `Condition`s, `AlertBook` (debounce, `alerts.json`), `Alerter`, `Watchdog` | alerts, health, gauges |
| `src/config.rs` | TOML structs (`deny_unknown_fields`), defaults, validation, `Secrets::from_env`, `secret_from_env`, `mcp_headers`, provider resolution (`provider_settings`, `profile_llm`) | any config key |
| `src/http.rs` | axum router: webhook, `/confirm/*`, `/healthz` (in-memory health + bounded ping), `/metrics` (refreshes health gauges) | routes |
| `src/webhook.rs` | webhook handler (bounded by `STORE_TIMEOUT`): version, signature, refuse while the DB is corrupt, dedupe, multipart, audio write (`io_permits`), delivery + job insert | webhook behaviour |
| `src/signature.rs` | HMAC-SHA256 signing string and constant-time verify | signature format |
| `src/db.rs` | SQLite schema (`MIGRATION_1`, `SCHEMA_VERSION`), queue (atomic claim, quarantine of unreadable jobs, requeue of orphaned `running` jobs), confirmations, audit log, retention; `DbStatus` (every statement's error class), `health_check` (fresh-connection `quick_check` + write probe), `queue_stats` | persistence, migrations, DB health |
| `src/worker.rs` | job loop: one task per job (`InFlight`, up to `worker_concurrency`), attempt deadline, panic isolation, max age, transcribe, run agent, retry with back-off, notify, audio retention | job lifecycle |
| `src/stt.rs` | whisper-server client and fallback chain | transcription |
| `src/audio.rs` | M4A (AAC-LC) to 16 kHz mono WAV in pure Rust (symphonia + hound); `Budget` (deadline and duration checked in the decode loop) and `Decoder` (size cap, semaphore, timeout; `with_fn` for tests) | audio decoding |
| `src/agent.rs` | provider-neutral agent loop, `AgentState`, tool discovery, planning, confirmations, tool execution | agent behaviour |
| `src/policy.rs` | `Policies` (allow/confirm/ask/deny/code; argument conditions `when`/`when_any`/`unless` as JSON pointer -> case-insensitive glob; `matched`, `always_denies`) and per-profile `ToolFilter` globs | tool policy rules |
| `src/authcode.rs` | spoken authorisation codes: `AuthCodes` (keyed HMAC tags, never plain codes; `scan` strips trigger + code into a marker, `scrub` strips code words), normalisation, `decide` (combine whisper and phone scans, lockout), `CodeAuth` ("code X until T") | code detection and redaction |
| `src/classifier.rs` | Shieldstral approval classifier: `Classifier` (request body, bounded `classify`, health reporting), `document`, `score_from_response`, `Verdict` | automatic approval |
| `src/tools.rs` | `ToolBackend` trait (`housekeeping`), `McpBackend` (rmcp client; sessions closed when idle `MCP_SESSION_IDLE` or dead, bounded close), `to_defs` (sorted, sanitised names), `check_schema` | MCP integration |
| `src/confirm.rs` | `GET /confirm/{token}` page, `POST /confirm/{token}/{approve,deny}` | confirmation UX |
| `src/notify.rs` | `Notifier` trait, ntfy client (`with_topic` for alerts), `Kind` (incl. `Alert`, `Recovered`), `LogNotifier` | notifications |
| `src/claude.rs` | thin Anthropic Messages client (headers, usage metrics); legacy re-exports | Anthropic HTTP details |
| `src/llm/mod.rs` | `Provider`, `ChatModel` trait, `Turn`/`StopKind`/`ToolCall`/`ToolResult`, `Llms`, `parse_arguments` | adding a provider |
| `src/llm/anthropic.rs` | Anthropic adapter: request body, cache breakpoints, fallbacks, `sanitize_fallback` | Claude request/response shape |
| `src/llm/chat.rs` | OpenAI-style adapter with a `Dialect` per provider (OpenAI, Mistral, OpenRouter, Ollama native) | other providers' shapes |
| `src/llm/transport.rs` | shared JSON POST with retries, quota detection, error formatting | retry rules |
| `src/eval/mod.rs` | `ringmaster eval`: `Options`, `ModelSpec` (`provider:model`, resolved like a profile), `Row` (one per request, `results.jsonl`), `Sink` (redacts every key), `run`, `offered_tools` (fixtures + distractors through `agent::prepare_tools`) | the eval CLI, results format |
| `src/eval/fixtures.rs` | fixture formats and checks: `Matcher`/`MatchSet` (policy-rule syntax, matched by `Policies::first_match`), `Script` (scripted tool results, `{{/pointer}}` templates), agent/capture/judge suites | fixture formats |
| `src/eval/suites.rs` | `Runner::converse` (production first message, adapters and error results, scripted tools), agent scoring (`score_call`, `turn_outcome`), `capture_outcome` (false success, junk cards), judge and classifier runs, cold/warm | scoring |
| `src/eval/judge.rs` | `JUDGE_SYSTEM_PROMPT` (the one place for a future LLM-judge mode), `judge_message` (the classifier's `document`), `parse_judge_reply` (first JSON object) | judge prompt |
| `src/eval/stats.rs`, `src/eval/report.rs` | AUC, best cut-off and margin, percentiles, title overlap; `report.md` | report |
| `eval/cases/` | synthetic fixtures: `tools.json`, `distractors.json`, `profiles.toml`, `results.toml`, `agent.toml`, `capture.toml`, `judge.toml` | eval cases |
| `tests/eval.rs` | eval against wiremock OpenAI, Ollama, Anthropic and classifier servers; fixture and personal-data checks | eval behaviour |
| `docs/model-evaluation.md` | user guide for `ringmaster eval` | eval docs |
| `tests/pipeline.rs` | end-to-end tests: signed webhook to notification, real rmcp server, wiremock model APIs | any behaviour change |
| `tests/fixtures/tone-1s.m4a` | real AAC-LC recording for decoder tests | |
| `deploy/docker-compose.example.yml` | reference deployment (ringmaster, cloudflared, MCP servers on an internal network) | deployment changes |
| `docs/mac-transcription.md` | whisper.cpp setup on the Mac | STT setup |
| `docs/mac-shieldstral.md` | llama.cpp `llama-server` with Shieldstral on the Mac | classifier setup |
| `config.example.toml` | complete commented example config | every config change |

## Setup and checks

- Use `cargo +stable`. `rust-version` is 1.89 (edition 2024) and some machines
  default to an older toolchain.
- Run tests with proxy variables unset; the tests talk to localhost servers:

  ```sh
  env -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY cargo +stable test --locked
  ```

- CI (`.github/workflows/ci.yml`) runs these jobs; their names are the
  required checks on `main`. Run all of them before pushing:

  | Check | Command |
  |---|---|
  | Format | `cargo +stable fmt --all --check` |
  | Clippy | `cargo +stable clippy --all-targets --locked -- -D warnings` |
  | Test | `cargo +stable test --locked` (proxies unset) |
  | Cargo deny | `cargo deny check` (CI: `check advisories bans licenses sources`) |
  | Docker build | `docker buildx build --platform linux/amd64,linux/arm64 .` |

- Also run `cargo +stable run --quiet -- check-config --config config.example.toml`;
  it must print `ok:` and exit 0.
- The Docker build cross-compiles with `tonistiigi/xx` (no QEMU) and needs a C
  compiler for bundled SQLite and aws-lc. It takes several minutes; it is
  worth running when you touch `Cargo.toml`, `Cargo.lock` or the `Dockerfile`.

## Invariants (do not break these)

1. **MCP and provider problems degrade; they never fail a job or stop the
   service.**
   - Discovery: every server the profile can use is listed concurrently with
     `agent.tool_discovery_timeout_secs`; a down, hung or erroring server is
     skipped and audited as `tool_server_skipped` (`Agent::start`, `src/agent.rs`).
   - Tools whose input schema the APIs would reject are dropped and audited as
     `tool_skipped` (`tools::check_schema`).
   - A permanent 400 on the first request while tools are offered is retried
     once without tools, unless it is about thinking/reasoning (`rejects_tools`
     and the `tools_dropped` branch in `Agent::run`).
   - The reply then ends with `Unavailable: <servers or "all tools">. You said:
     "<transcript>"` (`with_unavailable_note`).
   - A failing or timed-out tool call goes back to the model as an error
     result (`Agent::execute`). Bad `[[mcp_servers]]` entries are dropped with
     a warning at start-up (`Config::prune_mcp_servers`); `check-config` fails
     on them. Missing MCP credentials are warnings (`config::mcp_headers`).
   - API keys: only providers in use need one (`Config::validate_runtime`
     refuses to start without those); a client with no key fails only its own
     requests (`complete` returns a permanent error).
2. **Assistant messages are echoed back unchanged; history is append-only.**
   Each adapter stores the provider's own message format: Anthropic keeps the
   full `content`, thinking blocks included (only `sanitize_fallback` removes
   blocks of a declined attempt before a `fallback` block); `chat.rs` keeps a
   whitelist plus `Dialect::echo` fields (`reasoning_details`, Ollama
   `thinking`). Provider, model, effort, system prompt and tools are frozen
   in `AgentState` when the job starts and never change for that job.
3. **Prompt caching.** Volatile context (current time, recording time,
   gesture, transcript source, unavailable servers, transcript) goes only in
   the first user message (`agent::context_text`), never in the system prompt
   or tool definitions. Tool lists are sorted (`tools::to_defs`). Anthropic
   puts a cache breakpoint after the system prompt plus automatic caching of
   the tail (`Anthropic::body`).
4. **Never run a cut-off or malformed tool call.** `max_tokens` (Anthropic),
   `length`/`model_length` (chat) map to `StopKind::MaxTokens` and fail the
   job without running anything. Arguments that are not a JSON object
   (`llm::parse_arguments`) become `CallAction::Reject` with an error result
   asking the model to retry (`Agent::handle_tool_turn`).
5. **Tool policies** (`src/policy.rs`): globs over `server/tool`, `*` does not
   cross `/` (`literal_separator`), first matching rule wins, no match means
   `confirm`. A rule with argument conditions matches only when its tool glob
   and every condition match; a missing pointer never matches (for `unless`
   the rule then applies), lists match if any element does (`unless`: all).
   Only tools whose first applicable rule is an unconditional `deny`
   (`Policies::always_denies`) are filtered out before the model sees them
   (`Agent::start`); every call is checked again with its arguments
   (`Policies::matched` in `Agent::handle_tool_turn`), so a conditional
   `deny` is refused at call time with an error result. Profile `tools` globs
   are a second, independent allow-list (`ToolFilter`).
6. **Secrets come only from environment variables or `<NAME>_FILE`**
   (`config::secret_from_env`). Never add secret values to config structs,
   log lines, audit entries or `Debug` output (`Secrets` has a redacting
   `Debug`). Every credential header is a sensitive `HeaderValue`: LLM keys
   via `config::credential_header` (held as such by `ClaudeClient` and
   `ChatCompletions`), MCP headers in `McpBackend::new`, ntfy via
   `bearer_auth`. Error text built from a server's reply goes through
   `config::redact_credentials` (`transport::redact` for LLMs), which also
   removes every value registered with `config::register_secret` (spoken
   codes). The LLM
   `Transport` never follows redirects (`x-api-key` would follow one to
   another host). Request spans log the route template, not the path
   (confirmation tokens). `tests/log_redaction.rs` checks the logs.
7. **Persisted `AgentState` stays backwards compatible.** Jobs saved by older
   versions must still deserialise and resume: new fields get
   `#[serde(default)]` (see `provider`, `final_text`, `unavailable`), never
   rename or remove fields, and a state without `provider` means Anthropic
   (tests: `old_states_without_provider_are_anthropic`,
   `state_saved_before_providers_still_resumes`). A state that no longer
   decodes no longer blocks the queue (invariant 9), but the job is
   quarantined and the user's request is dropped, so this still matters.
   Schema changes go in a new migration with a bumped `SCHEMA_VERSION`;
   never edit `MIGRATION_1` or `MIGRATION_2` (codes: `deliveries.phone_code`,
   `jobs.code_auth`, `auth_code_failures`, `auth_code_lockouts`; one
   transaction including the `user_version` bump).
8. **Webhook verification and dedupe** (`src/webhook.rs`, `src/signature.rs`):
   - Version header must be `1`; the HMAC covers
     `v1\n{timestamp}\n{delivery_id}\n{trigger}\n{1|0}\n` + raw body, is
     verified in constant time (`verify_slice`) before the body is parsed,
     and the timestamp must be within `timestamp_tolerance_secs` (300).
   - With a secret set, a request carrying signature headers is verified
     even when `require_signature = false`. `require_signature = true` with
     no secret refuses to start (`validate_runtime`); secrets are at least
     32 characters.
   - A delivery id already seen gets `200 {"status":"duplicate"}` and is not
     processed again (checked before parsing, and again by `INSERT OR IGNORE`
     inside the delivery + job transaction in `Db::insert_delivery`).
     Accepted jobs get `202`. Test events are recorded and never run the agent.
   - Requests the app should not retry get 4xx; storage problems get 503 so
     the app retries.

9. **One job never holds up the queue** (`src/worker.rs`, `src/db.rs`).
   - Each claimed job runs in its own task; up to `agent.worker_concurrency`
     (default 2, max 16) at once. The loop keeps claiming while a slot is
     free and never awaits a single job. Jobs may finish out of order.
   - Claiming is one `UPDATE ... RETURNING` in an `IMMEDIATE` transaction
     and skips jobs this process already has in flight, so no job is ever
     run by two tasks (`concurrent_claims_never_take_the_same_job`).
   - A job whose row no longer loads (`load_job` returns
     `LoadError::Unreadable`: bad `AgentState` JSON, a column of the wrong
     type, a missing delivery) is quarantined inside the claim transaction:
     `failed`, `last_error` "quarantined, not retried: ...", a
     `job_quarantined` audit row, and a notification carrying the transcript
     or phone transcription. Claiming then moves on. Database errors
     (`LoadError::Db`) still roll back and retry.
   - Each attempt (`Worker::attempt`) is bounded by
     `agent.attempt_timeout_secs` (default 900); running over is a transient
     error, so the normal back-off, "delayed" notification and max-age expiry
     apply.
   - A panic in a job is caught (`catch_unwind` in `Worker::handle`) and
     fails that job only. Jobs left `running` with no task (restart, crash,
     a lost task) are requeued by `Db::requeue_running` on every
     housekeeping pass, skipping the jobs in flight. This assumes one
     Ringmaster process per database.
   - Started jobs are not resumed more than `retry_max_secs` +
     `confirmation_ttl_secs` past their max age (unless the answer is already
     in `final_text`), so a job that keeps crashing the service, or comes
     back after a long outage, is stopped rather than retried for ever.
     Transient failures stop at the max age (`retry_later`); a decoder panic
     falls back to the phone transcription instead of retrying.
   - Jobs in `awaiting_confirmation` are never claimed; they take no slot.

10. **Problems are alerted, health is honest, and every task ends**
    (`src/health.rs`, `src/db.rs`, `src/lib.rs`).
    - Database errors are classified in `Db::call` (`DbStatus`):
      corruption (`SQLITE_CORRUPT`/`NOTADB`, or `quick_check` problems on a
      fresh connection) and storage errors (full, read-only, I/O) alert at
      once and stay set until a clean `health_check` (quick_check plus write
      probe); other failures alert after `alerts.db_error_threshold` in a
      row. While the DB is corrupt the webhook answers 503 before storing.
    - The `Watchdog` is its own task: heartbeats (worker loop, watchdog),
      due jobs (`queue_stats`), jobs past `attempt_timeout_secs` + margin,
      failure bursts, blocking work past its limit. Each step is bounded
      (DB check 30 s and never two at once, queue query 10 s, each send 20 s).
    - `Alerter`/`AlertBook`: one alert per condition start, reminders every
      `alerts.reminder_secs`, one "recovered"; state in `alerts.json`. An
      alert that fails to send is logged and retried; sending never blocks
      or fails anything. Alert text carries counts, job ids, ages and
      `DbErrorClass` names only: never a transcript, model output, SQLite
      message text or secret (`alerts_never_carry_transcripts_or_secrets`).
    - `/healthz` is non-200 only when recordings cannot be stored or
      processed (DB corrupt/storage/unresponsive, worker or watchdog
      stalled). It does no heavy work. Do not add conditions a restart
      cannot fix and that are not the container's fault.
    - Every spawned task or thread ends: job tasks by the attempt deadline
      and the shutdown grace; worker, watchdog and HTTP by `run_service`;
      `Db::call` takes the async lock before `spawn_blocking` (one blocking
      thread per hung disk, not one per caller); decodes are bounded by
      `Budget` and capped by `stt.decode_concurrency`; webhook audio writes
      by `io_permits` and `STORE_TIMEOUT`; MCP sessions by idle/dead close;
      the runtime is dropped with `shutdown_timeout`. A new `spawn`/
      `spawn_blocking` needs a bound like these, and blocking work should be
      registered with `Health::blocking` so the watchdog sees it stuck.

11. **The approval classifier can approve but never deny, and fails over to
    the human** (`src/classifier.rs`, `Agent::judge` and
    `Agent::handle_tool_turn` in `src/agent.rs`).
    - Only `confirm` calls are classified, and only in `shadow` or
      `classifier` mode (`approval.mode`, profile `approval_mode`; default
      `manual` = no classifier at all). `deny` calls are never classified;
      `ask` calls always go to the human and are never sent to the
      classifier (nor, in classifier mode, are the other calls of their
      turn). `code` calls are never classified: they run on a valid code
      (`CallAction::Authorised`), are refused, or (`on_missing = "ask"`) go
      to the human as `ask`.
    - A turn runs without asking only when every `confirm` call in it scored
      below `approval.auto_approve_below` (`Judged::Safe`); anything else
      parks the turn for Approve/Deny exactly as in manual mode. No verdict
      (down, timeout, error status, no `yes`/`no` logprobs, a call too long
      to send whole) is `Judged::Unavailable` and asks the human: it never
      runs a call, never fails a job and never denies.
    - `classify` is bounded by `approval.request_timeout_secs` (retries
      included); calls of a turn are judged concurrently.
    - Every classification is audited (`classification`), counted
      (`ringmaster_classifications_total`, `ringmaster_classifier_score`),
      and every auto-approved call is kept in `AgentState::auto_approved`
      and listed at the end of the job's final text
      (`with_auto_approved_note`). Repeated failures raise
      `Condition::ClassifierFailing`; it never affects `/healthz`.
    - New persisted fields (`AgentState::auto_approved`,
      `PendingTurn::approval_mode`/`judgements`) are `serde(default)` and
      skipped when empty, so manual-mode states keep their old shape.

12. **Authorisation codes are checked by Ringmaster, never seen by the LLM,
    the classifier, ntfy or the logs** (`src/authcode.rs`,
    `Worker::check_codes`, the webhook, `Agent::handle_tool_turn`).
    - Code values come only from the environment (`code_env`, `_FILE`);
      `AuthCodes` holds keyed HMAC tags, compares with `verify_slice`
      (constant time), and has a redacting `Debug`. A configured code
      without a value refuses to start.
    - The phone's transcription is scanned in the webhook before it is
      stored, logged or audited; the whisper transcript in
      `Worker::check_codes` before the `transcript` audit row and
      `save_transcript_with_codes`. Trigger + code become a marker
      (`[authorisation code accepted]`/`rejected`/locked), code words alone
      become `[redacted]`. Everything downstream (`jobs.transcript`, audit,
      `AgentState`, model and classifier requests, confirmation summaries,
      notifications) only ever sees the stripped text; `Worker::notify` and
      confirmation summaries `scrub` again. Without `[auth_codes.codes]`
      nothing is scanned or changed.
    - A recording that held a code is deleted right after transcription
      (`Worker::delete_code_audio`, audit `audio_deleted`), whatever the
      retention.
    - The job stores only `CodeAuth` ("code X until T", T = recording time +
      `valid_for_secs`), saved in the same transaction as the transcript and
      the wrong-code count. A code authorises `code` calls naming it in that
      job only; `execute` re-checks expiry and lockout before running.
    - Wrong codes are counted in `auth_code_failures`; too many start a
      lockout (`auth_code_lockouts`, persisted) during which codes are not
      checked, and the watchdog raises `Condition::AuthCodeLockout` (counts
      only). A transcript with any wrong attempt authorises nothing.
    - Tests: `spoken_code_authorises_the_disarm_without_ntfy_or_classifier`
      and its neighbours in `tests/pipeline.rs` assert the code is absent from
      the database dump, model and classifier requests and notifications;
      `tests/log_redaction.rs` checks the logs.

13. **The eval uses production request code, and its fixtures stay
    synthetic** (`src/eval/`, `eval/cases/`).
    - Every eval request goes through the real adapters (`Llms`,
      `ChatModel::complete`, `Classifier::classify`) with the production
      system prompt (`agent::system_prompt`), first message
      (`agent::context_text`), tool list (`agent::prepare_tools`: schema
      check, profile filter, `to_defs` sorting) and error results
      (`agent::unknown_tool_result`, `agent::malformed_arguments_result`).
      Change those helpers rather than copying their logic into the eval, so
      results keep reflecting what the service does. Known differences: no
      MCP servers (fixture tools, scripted results), no tool policies,
      confirmations or classifier approval in the loop, no retry without
      tools after a 400.
    - Expectations use the policy matcher (`Policies::first_match`), so they
      are written like `[[tool_policies]]` rules.
    - Keys come from the environment as in production and never reach
      `results.jsonl` or `report.md` (`Sink` redacts every line;
      `agent_suite_scores_calls_through_the_openai_adapter` checks it).
    - Fixtures are invented: no real devices, entity ids, boards, names,
      places, transcripts, IP or e-mail addresses
      (`fixtures_hold_no_obvious_personal_data`). Everything in them is sent
      to the providers being evaluated.
    - CI never calls a real provider: `tests/eval.rs` uses wiremock.

Other behaviour to keep: `GET /confirm/*` never changes state; confirmation
tokens are 256-bit, stored hashed (`hash_token`), single use, and expire after
`confirmation_ttl_secs` (no answer counts as deny). Jobs older than
`job_max_age_secs` are not started. Tool calls are at-least-once.

## Testing conventions

- Mock every external API with `wiremock`. Never call a real provider: there
  are no API keys in CI or locally.
- End-to-end tests go in `tests/pipeline.rs`. Use `harness(turns, whisper_text)`
  or `harness_with(turns, whisper_text, |home| servers)`; `harness_config`
  takes a whole config and `config_with(llm)` swaps the `[claude]` section for
  another provider. The live MCP server is a real rmcp server (`start_mcp`,
  server name `home`).
- Script model replies with `ScriptedClaude` (one mock for every provider's
  path) using `tool_use_turn`/`final_turn` (Anthropic), `chat_tool_turn`/
  `chat_final` (OpenAI-style) or `ollama_tool_turn`/`ollama_final`. Script an
  API error with `{"__status": 400, "__body": {...}}`. Inspect request bodies
  via `claude_seen` and paths/headers via `meta(&h)`.
- Drive the flow with `signed_request`, `send`, `tick` and `run_command`.
  `tick` processes every job due when it starts (concurrently, up to
  `worker_concurrency`) and waits for all of them; retries scheduled during
  the tick wait for the next one. Script order is per request, so tests that
  depend on which job gets which turn set `worker_concurrency = 1`
  (`config_agent`), or match on the request body
  (`hang_requests_containing`). `queue_command` queues a job with a phone
  transcription; `harness_full` wraps the tool backend (see
  `PanickingTools`).
- Health and alerts: the harness has `health` and a `watchdog` that sends
  to the same recorder. Call `h.watchdog.check(t)` with an explicit time
  (move heartbeats with `h.health.beat_worker`) instead of sleeping; read
  alerts with `alerts(&h)`, gauges with `gauges(&h)` (a local Prometheus
  recorder, so parallel tests do not share values) and `/healthz` with
  `healthz(&h)`. `harness_decoder` swaps in a misbehaving decoder;
  `set_read_only` makes the database file read-only.
- **wiremock pooling gotcha:** a dropped `MockServer` goes back to a pool and
  is handed to tests running at the same time, so it is not "unreachable".
  For a dead endpoint, bind `std::net::TcpListener` to `127.0.0.1:0`, take
  the port and drop the listener, as `dead_server()` in `tests/pipeline.rs`
  and `unreachable_server_is_transient` in `src/stt.rs` do.
- Unit tests live next to the code in `#[cfg(test)] mod tests`.
- New behaviour needs a test that fails without the change.

## Docs to update with any user-visible change

- README: the configuration reference table, the secrets table, provider
  notes, endpoints, metrics and audit kinds as relevant.
- `config.example.toml`: commented examples of new keys. It must still pass
  `check-config`.
- `deploy/docker-compose.example.yml`: new environment variables, secrets,
  volumes or services.
- `docs/mac-transcription.md` for anything about the whisper-server contract,
  `docs/mac-shieldstral.md` for the classifier server contract.
- `docs/model-evaluation.md` for anything about `ringmaster eval`, its
  fixtures or its report.

## Contributing and releases

- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, ...). PR titles
  become the squash commit, so they must follow the convention too.
- `main` is protected: squash merges only, required checks Format, Clippy,
  Test, Cargo deny and Docker build. Branch, open a PR, wait for green.
- release-please (`.github/workflows/release.yml`, `release-please-config.json`)
  keeps a release PR open. Merging it tags `vX.Y.Z`, creates the GitHub release
  and pushes `ghcr.io/lazzurs/ringmaster:X.Y.Z`, `X.Y` and `latest` for
  linux/amd64 and linux/arm64 with SBOM and provenance.
- Release PR CI: without a `RELEASE_PLEASE_TOKEN` secret, release-please
  opens its PR with `GITHUB_TOKEN`, which starts no workflows. The "Run CI on
  the release PR" step dispatches `ci.yml` on the release branch, but those
  `workflow_dispatch` runs are not linked to the PR and do not satisfy branch
  protection. A human closing and reopening the release PR triggers the
  normal `pull_request` CI. Adding `RELEASE_PLEASE_TOKEN` (fine-grained PAT or
  GitHub App token with contents + pull-requests write) fixes it for good.
- Before 1.0 (`bump-minor-pre-major`, `bump-patch-for-minor-pre-major`), `feat`
  bumps the patch version and breaking changes bump the minor.
- Do not edit `CHANGELOG.md` or `.release-please-manifest.json` by hand.
- Do not add dependencies lightly. Each one must pass `cargo deny` (licences,
  advisories, bans, sources); the licence allow list in `deny.toml` is
  limited to GPL-3.0-compatible licences. The runtime image is distroless
  (glibc, no shell), so a dependency that needs other shared libraries at run
  time will not work there; C code must build under the `Dockerfile`'s
  cross-compiling builder.

## Licence

GPL-3.0-or-later (`LICENSE`, `Cargo.toml`). Copyright (C) 2026 Rob Lazzurs.

## Security notes

- The only public surface is `POST /webhook` and `/confirm/*`. In the
  reference deployment, the tunnel routes only `^/(webhook|confirm/.*)$`;
  `/metrics` and `/healthz` stay private. Do not add public routes without
  updating the deployment docs.
- The transcript is untrusted input; profiles and policies are the safety
  boundary, not the prompt.
- MCP servers stay on a private network; Ringmaster is the only MCP client and
  runs every tool call itself.
- The container runs as uid 65532 with a read-only root filesystem; `/data`
  is the only writable path.
