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
  -> MCP tools over streamable HTTP, gated by allow/confirm/deny policies
  -> ntfy notification (result, problem, or Approve/Deny buttons)
```

Jobs are durable: the conversation is saved after every step, so restarts,
outages and pending confirmations resume where they stopped.

## Code map

| Path | What lives there | Change it for |
|---|---|---|
| `src/main.rs` | CLI: `serve`, `healthcheck`, `check-config`; logging setup | new subcommands, `check-config` output |
| `src/lib.rs` | `serve()`: wires config, secrets, DB, notifier, LLMs, MCP backend, worker, HTTP; graceful shutdown | start-up wiring |
| `src/config.rs` | TOML structs (`deny_unknown_fields`), defaults, validation, `Secrets::from_env`, `secret_from_env`, `mcp_headers`, provider resolution (`provider_settings`, `profile_llm`) | any config key |
| `src/http.rs` | axum router: webhook, `/confirm/*`, `/healthz`, `/metrics` | routes |
| `src/webhook.rs` | webhook handler: version, signature, dedupe, multipart, audio write, delivery + job insert | webhook behaviour |
| `src/signature.rs` | HMAC-SHA256 signing string and constant-time verify | signature format |
| `src/db.rs` | SQLite schema (`MIGRATION_1`, `SCHEMA_VERSION`), queue (atomic claim, quarantine of unreadable jobs, requeue of orphaned `running` jobs), confirmations, audit log, retention | persistence, migrations |
| `src/worker.rs` | job loop: one task per job (`InFlight`, up to `worker_concurrency`), attempt deadline, panic isolation, max age, transcribe, run agent, retry with back-off, notify, audio retention | job lifecycle |
| `src/stt.rs` | whisper-server client and fallback chain | transcription |
| `src/audio.rs` | M4A (AAC-LC) to 16 kHz mono WAV in pure Rust (symphonia + hound) | audio decoding |
| `src/agent.rs` | provider-neutral agent loop, `AgentState`, tool discovery, planning, confirmations, tool execution | agent behaviour |
| `src/policy.rs` | `Policies` (allow/confirm/deny) and per-profile `ToolFilter` globs | tool policy rules |
| `src/tools.rs` | `ToolBackend` trait, `McpBackend` (rmcp client), `to_defs` (sorted, sanitised names), `check_schema` | MCP integration |
| `src/confirm.rs` | `GET /confirm/{token}` page, `POST /confirm/{token}/{approve,deny}` | confirmation UX |
| `src/notify.rs` | `Notifier` trait, ntfy client, `LogNotifier` | notifications |
| `src/claude.rs` | thin Anthropic Messages client (headers, usage metrics); legacy re-exports | Anthropic HTTP details |
| `src/llm/mod.rs` | `Provider`, `ChatModel` trait, `Turn`/`StopKind`/`ToolCall`/`ToolResult`, `Llms`, `parse_arguments` | adding a provider |
| `src/llm/anthropic.rs` | Anthropic adapter: request body, cache breakpoints, fallbacks, `sanitize_fallback` | Claude request/response shape |
| `src/llm/chat.rs` | OpenAI-style adapter with a `Dialect` per provider (OpenAI, Mistral, OpenRouter, Ollama native) | other providers' shapes |
| `src/llm/transport.rs` | shared JSON POST with retries, quota detection, error formatting | retry rules |
| `tests/pipeline.rs` | end-to-end tests: signed webhook to notification, real rmcp server, wiremock model APIs | any behaviour change |
| `tests/fixtures/tone-1s.m4a` | real AAC-LC recording for decoder tests | |
| `deploy/docker-compose.example.yml` | reference deployment (ringmaster, cloudflared, MCP servers on an internal network) | deployment changes |
| `docs/mac-transcription.md` | whisper.cpp setup on the Mac | STT setup |
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
   the first user message (`Agent::context_text`), never in the system prompt
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
   `confirm`. `deny` tools are filtered out before the model sees them
   (`Agent::start`) and rejected again if called. Profile `tools` globs are a
   second, independent allow-list (`ToolFilter`).
6. **Secrets come only from environment variables or `<NAME>_FILE`**
   (`config::secret_from_env`). Never add secret values to config structs,
   log lines, audit entries or `Debug` output (`Secrets` has a redacting
   `Debug`). MCP credential headers are marked sensitive
   (`HeaderValue::set_sensitive` in `McpBackend::new`); ntfy uses
   `bearer_auth`. Note: LLM key headers built in `src/claude.rs` and
   `src/llm/chat.rs` are passed as plain strings to `Transport`, which does
   not mark them sensitive; do so if you touch that code.
7. **Persisted `AgentState` stays backwards compatible.** Jobs saved by older
   versions must still deserialise and resume: new fields get
   `#[serde(default)]` (see `provider`, `final_text`, `unavailable`), never
   rename or remove fields, and a state without `provider` means Anthropic
   (tests: `old_states_without_provider_are_anthropic`,
   `state_saved_before_providers_still_resumes`). A state that no longer
   decodes no longer blocks the queue (invariant 9), but the job is
   quarantined and the user's request is dropped, so this still matters.
   Schema changes go in a new migration with a bumped `SCHEMA_VERSION`;
   never edit `MIGRATION_1`.
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
- `docs/mac-transcription.md` for anything about the whisper-server contract.

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
