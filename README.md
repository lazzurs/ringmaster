# Ringmaster

Voice commands from a smart ring. Hold the ring's button, speak, let go:
the phone app posts the recording to Ringmaster, which transcribes it on a Mac
with whisper.cpp, hands the text to Claude with tools from your MCP servers
(smart home, Nextcloud Deck, ...), and sends the outcome back to your phone
with ntfy. Risky actions wait for you to tap **Approve**.

It is one Rust service in one container, built for a home server running
Docker Compose.

```
 ring ──BLE──► phone app ──HTTPS webhook──► Cloudflare Tunnel ──► ringmaster ─┬─► SQLite (/data)
                                                                              │
          ┌───────────────────────── worker ──────────────────────────────────┘
          │
          ├─► whisper-server on the Mac (LAN)   M4A → WAV in Rust, POST /inference
          │     └─ fallback Mac, then the phone's own transcription
          ├─► Claude Messages API               agent loop, tools as client tools
          │     └─► MCP servers (streamable HTTP, private compose network)
          └─► ntfy                              result / error / Approve-Deny
```

## How a recording is handled

1. **Webhook** (`POST /webhook`). The raw body is read first; the version
   header is checked, the HMAC signature verified in constant time with a
   5-minute timestamp window, the delivery id deduplicated, the multipart body
   parsed, the audio written to `/data/audio` and the job inserted in the same
   SQLite transaction. The app gets `202` in milliseconds; nothing here waits
   for the Mac or Claude. A repeated delivery id gets `200` and is not
   processed again. Test events are logged (and optionally notified) but never
   reach the agent.
2. **Transcription**. The worker decodes the AAC-LC M4A to 16 kHz mono WAV in
   pure Rust (symphonia + hound; no ffmpeg anywhere) and calls whisper.cpp's
   `/inference` with an initial prompt built from your vocabulary. Primary
   server, then fallback server, then the phone's own transcription. If no
   server answers and there is no phone transcription, the job is retried
   with back-off.
3. **Agent**. The gesture (`X-Index-Trigger`) selects a **profile**: a system
   prompt plus an allow-list of tools. Ringmaster is the MCP client: it lists
   the tools of the servers the profile can use, freezes them (sorted) into
   the job, and runs a manual tool loop against the Messages API, executing
   tool calls itself so the MCP servers never leave the private network.
4. **Policies**. Every tool call is checked against `tool_policies`:
   `allow` runs, `deny` never runs (and is not even offered to Claude),
   `confirm` pauses the job and sends an ntfy notification with Approve/Deny
   buttons. Tools that no rule matches need confirmation.
5. **Result**. The final answer, any error, and any delay are sent through
   ntfy. Everything is in the audit log.

Jobs are durable: a restart, a sleeping Mac, an API outage or a pending
confirmation all resume from the last saved step.

## Claude API usage

* Raw HTTPS with `reqwest`, `anthropic-version: 2026-10-21`.
* Model is configurable: `claude-opus-5-5` (default), `claude-sonnet-5-5`, or
  `claude-haiku-5-5` (cheapest and fastest; fine for simple commands). Set it
  globally in `[claude] model` or per profile.
* `output_config.effort` is always sent (default `low`: these are short
  commands). No `thinking` parameter (adaptive thinking is the default and
  cannot be turned off on Opus 5.5), no `budget_tokens`, no sampling
  parameters, no prefill, `tool_choice` left at `auto`.
* The full assistant `content`, thinking blocks included, is appended
  unchanged each turn; all tool results of a turn go back in one user message
  (`is_error: true` for failures, denials and timeouts). `stop_reason` is
  checked before content is read: `refusal` is reported, `max_tokens` never
  runs a possibly truncated tool call, `pause_turn` is resumed.
* Server-side refusal fallbacks (`fallbacks: "default"` with
  `anthropic-beta: server-side-fallback-2026-07-01`) are on for Opus and
  Sonnet. Haiku 5.5 has no server-side fallback and rejects the list, so
  `refusal_fallbacks = "auto"` leaves it off there.
* Prompt caching: the tool list (sorted) and the system prompt are frozen per
  job with a cache breakpoint after the system prompt, plus automatic caching
  for the conversation tail. The current time, gesture and transcript go in
  the first user message, never in the system prompt.
* 429, 5xx, 529 and connection errors are retried with exponential back-off,
  honouring `retry-after`; other 4xx are not. Each job is capped at
  `max_iterations` API calls (default 8).

## Configuration

`config.toml` (path from `--config` or `RINGMASTER_CONFIG`, default
`/etc/ringmaster/config.toml`). See [`config.example.toml`](config.example.toml)
for a complete, commented example. `ringmaster check-config` validates a file.

### Secrets (environment only)

Each can also be given as `<NAME>_FILE` pointing at a file.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (required) |
| `RINGMASTER_WEBHOOK_SECRET` | the ring app's signing secret, at least 32 characters (required unless `webhook.require_signature = false`) |
| `NTFY_TOKEN` | ntfy access token, if the topic needs one |
| any `token_env` named by an `[[mcp_servers]]` entry | bearer token for that MCP server |

### Reference

| Key | Default | Meaning |
|---|---|---|
| `server.listen` | `0.0.0.0:8080` | HTTP listen address |
| `server.public_base_url` | none | external URL (tunnel hostname) for the Approve/Deny links; without it confirmations have no buttons |
| `server.data_dir` | `/data` | SQLite database and recordings |
| `server.max_body_bytes` | 25 MiB | webhook body limit |
| `webhook.path` | `/webhook` | webhook path |
| `webhook.require_signature` | `true` | reject unsigned requests |
| `webhook.timestamp_tolerance_secs` | `300` | allowed clock skew |
| `webhook.notify_test_events` | `true` | send a notification for test events |
| `stt.primary_url` / `stt.fallback_url` | none | whisper-server base URLs |
| `stt.inference_path` | `/inference` | |
| `stt.language` | `en` | `auto` to detect |
| `stt.vocabulary` | `[]` | names to bias recognition with |
| `stt.prompt` | built from vocabulary | explicit initial prompt |
| `stt.use_phone_transcription` | `true` | last-resort fallback |
| `stt.connect_timeout_secs` / `stt.timeout_secs` | `5` / `120` | |
| `claude.api_base` | `https://api.anthropic.com` | |
| `claude.model` | `claude-opus-5-5` | or `claude-sonnet-5-5`, `claude-haiku-5-5` |
| `claude.effort` | `low` | `low` `medium` `high` `xhigh` `max` |
| `claude.max_tokens` | `16000` | covers thinking and the reply |
| `claude.max_iterations` | `8` | API calls per job |
| `claude.refusal_fallbacks` | `auto` | `auto` `on` `off` |
| `claude.request_timeout_secs` / `claude.max_retries` | `120` / `4` | |
| `agent.timezone` | `UTC` | IANA zone for the time given to Claude |
| `agent.base_system_prompt` | built in | replaces the shared preamble |
| `agent.confirmation_ttl_secs` | `600` | Approve/Deny link lifetime; no answer counts as deny |
| `agent.job_max_age_secs` | `3600` | give up (and say so) after this; profiles override |
| `agent.retry_base_secs` / `agent.retry_max_secs` | `15` / `600` | job back-off |
| `agent.notify_on_retry` | `true` | one "delayed" notification per job |
| `agent.tool_timeout_secs` | `60` | per MCP call |
| `agent.max_tool_result_chars` | `20000` | tool output truncation |
| `retention.audio_days` | `30` | delete recordings after this (0 keeps them) |
| `notify.ntfy.url` / `.topic` | `https://ntfy.sh` / required | |
| `notify.ntfy.action_headers` | `{}` | headers the phone sends with Approve/Deny |
| `[[mcp_servers]]` `name`, `url`, `token_env` | | streamable HTTP MCP servers |
| `[[tool_policies]]` `match`, `policy` | | glob over `server/tool`; `allow` `confirm` `deny`; first match wins |
| `[profiles.<name>]` `triggers`, `system_prompt`, `tools` | | gesture → prompt + tool allow-list (globs; `*/*` = all) |
| `[profiles.<name>]` `model`, `effort`, `max_iterations`, `job_max_age_secs` | | per-profile overrides |

Glob `*` does not cross the `/` between server and tool: `homey/*` matches
every Homey tool, `*` alone matches nothing.

A profile's servers are required or optional depending on how its `tools`
name them. A server named literally (`nextcloud/deck_*`) is required: if it is
down, the job waits and retries. A server reached only through a wildcard
(`*/*`) is optional: if it is down, the job runs without its tools, Claude is
told which servers are unavailable, and the skip is recorded in the audit log
as `tool_server_skipped`. Adding a new MCP server therefore cannot break
existing gestures.

## Phone app setup

In the ring's companion app, open the webhook settings for each gesture:

| Gesture | `X-Index-Trigger` | Profile in the example config |
|---|---|---|
| single click and hold | `single-click-hold` | `capture`: add a card to the Deck Inbox, nothing else |
| double click and hold | `double-click-hold` | `command`: smart home and Nextcloud, with confirmations |

For each gesture:

* **URL**: `https://<tunnel hostname>/webhook`
* **Delivery mode**: **Both** (recording and transcription) to start with.
  Ringmaster uses the Mac's transcription and keeps the phone's as a fallback
  for when the Mac is asleep. "Recording only" also works.
* **Signing**: on, with a secret of at least 32 random bytes, e.g.
  `openssl rand -hex 32`. The same value goes in `RINGMASTER_WEBHOOK_SECRET`.
  (Use a separate secret per endpoint if you run more than one.)
* **Custom headers**: `CF-Access-Client-Id` and `CF-Access-Client-Secret` from
  the Cloudflare Access service token (below).
* Send a **test event**: Ringmaster answers `200`, logs it, and (by default)
  sends a "test event" notification. Test events never run the agent.

The app retries failed deliveries (network errors, 408, 429, 5xx) every
minute up to 10 times with the same delivery id; Ringmaster acknowledges
repeats without reprocessing them. Ringmaster answers `4xx` for requests it
will never accept (bad signature, unknown gesture), which the app does not
retry.

## Cloudflare Tunnel and Access

* Create a tunnel and run `cloudflared` in the compose file with its token.
  Public hostname → `http://ringmaster:8080`, **path** `^/(webhook|confirm/.*)$`
  so `/metrics` and `/healthz` are not exposed.
* Create an Access application for `<hostname>/webhook` with a **Service
  Auth** policy and a service token. The phone sends the token as the two
  custom headers above; requests without it never reach Ringmaster.
* `/confirm/*` is called by the ntfy app on your phone when you tap
  Approve/Deny. Either add an Access application for `<hostname>/confirm`
  with a **Bypass** policy (the links carry one-time 256-bit tokens that
  expire after `confirmation_ttl_secs`), or keep it behind the service token
  and put the two headers in `[notify.ntfy.action_headers]` (they are then
  stored on the ntfy server with every confirmation message).

## ntfy

Use a self-hosted ntfy, or ntfy.sh with a reserved topic and an access token.
Anyone who can read the topic sees your commands and could tap Approve, so
do not use an open public topic. Notifications:

| Kind | Priority | When |
|---|---|---|
| result | 3 | the agent finished; the body is Claude's one-line answer |
| approve? | 5 | a `confirm` tool is waiting; `Approve` / `Deny` buttons (`http` actions that `POST` to `/confirm/<token>/approve|deny`) and a `Details` link |
| problem | 4 | errors, refusals, give-ups, and the first delay of a job |
| info | 2 | test events, empty recordings |

## Transcription on the Mac

See [docs/mac-transcription.md](docs/mac-transcription.md): whisper.cpp built
from source (Homebrew's formula does not include the server), the
`large-v3-turbo` model, a LaunchAgent with `KeepAlive`, automatic login, and
`pmset` so the Mac never sleeps and restarts after a power cut.

## Deployment

[`deploy/docker-compose.example.yml`](deploy/docker-compose.example.yml) runs
Ringmaster, `cloudflared`, the Nextcloud MCP server
([cbcoutinho/nextcloud-mcp-server](https://github.com/cbcoutinho/nextcloud-mcp-server),
streamable HTTP on `:8000/mcp`, single-user app-password mode, Deck only) and
a placeholder for a Homey MCP server. MCP servers sit on an `internal`
network with no published ports.

The image runs as uid 65532 with a read-only root filesystem; `/data` (a
volume) is the only writable path. Bind-mounting a host directory instead of
a named volume needs `chown 65532:65532`.

### Endpoints

| Path | Purpose |
|---|---|
| `POST /webhook` | ring app deliveries |
| `GET /confirm/<token>` | shows a pending action with Approve/Deny buttons (GET changes nothing) |
| `POST /confirm/<token>/approve`, `/deny` | records the decision once |
| `GET /healthz` | `200 ok` when the database answers |
| `GET /metrics` | Prometheus metrics |

`ringmaster healthcheck` probes `/healthz` for the Docker `HEALTHCHECK`
(the image has no shell or curl).

### Logs and metrics

Logs are JSON on stdout (`RUST_LOG` sets levels; `RINGMASTER_LOG_FORMAT=text`
for humans). Metrics include `ringmaster_webhook_requests_total{result}`,
`ringmaster_jobs_total{outcome}`, `ringmaster_queue_depth`,
`ringmaster_stt_requests_total{source,result}`,
`ringmaster_claude_requests_total{status}`,
`ringmaster_claude_tokens_total{kind}` (input, output, cache_read,
cache_write), `ringmaster_tool_calls_total{outcome}`,
`ringmaster_confirmations_total{decision}` and
`ringmaster_notifications_total{status}`.

### Audit log

`/data/ringmaster.db` (SQLite, WAL). `deliveries` has every webhook delivery;
`jobs` the queue with transcript, transcript source and the stored
conversation; `confirmations` the hashed one-time tokens and decisions;
`audit` a timeline per job: `delivery`, `transcript`, `agent_start`,
`claude_response` (stop reason, tool calls, usage), `tool_call` (input,
policy action, result), `confirmation_requested` / `_answered` / `_decision`,
`retry_scheduled`, `job_finished` / `job_failed`.

The image has no shell; copy the database out to query it:

```sh
docker compose cp ringmaster:/data/ringmaster.db /tmp/ && sqlite3 /tmp/ringmaster.db \
  "select datetime(ts,'unixepoch'), kind, detail from audit order by id desc limit 20"
```

## Safety notes

* Profiles limit what each gesture can touch; policies decide what needs a
  tap. Start strict (`confirm` by default) and loosen per tool.
* Jobs older than `job_max_age_secs` are not started: a "turn the heating
  off" should not run hours later because the Mac was asleep. The example
  uses 15 minutes for commands and a day for captures.
* MCP tool calls are at-least-once: if Ringmaster dies after a tool ran but
  before the result was saved, the call runs again on restart.
* The transcript is untrusted input. The capture profile only has Deck
  tools, so even "unlock the door" said in capture mode becomes a card.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked          # unset http(s)_proxy/ALL_PROXY if your shell sets them
cargo deny check
docker buildx build --platform linux/amd64,linux/arm64 .
```

The tests cover HMAC verification against an independently computed vector,
dedupe, multipart parsing, M4A→WAV decoding of a real AAC-LC file
(`tests/fixtures/tone-1s.m4a`), the STT fallback chain, Claude retries, and
the whole pipeline end to end against a mocked Claude API (wiremock) and a
real MCP server (rmcp) on localhost, including approve and deny.

## Releases

Conventional Commits drive [release-please](https://github.com/googleapis/release-please):
it keeps a release PR open; merging it tags `vX.Y.Z`, publishes the GitHub
release with notes, and pushes `ghcr.io/lazzurs/ringmaster:X.Y.Z`, `X.Y` and
`latest` for linux/amd64 and linux/arm64 with SBOM and provenance
attestations. See `.github/workflows/release.yml` for how the release PR gets
its required checks.

## Licence

MIT. Not affiliated with or endorsed by the maker of the ring or its app.
