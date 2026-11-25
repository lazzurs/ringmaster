# Ringmaster

Voice commands from a smart ring. Hold the ring's button, speak, let go:
the phone app posts the recording to Ringmaster, which transcribes it on a Mac
with whisper.cpp, hands the text to an LLM (Claude by default; OpenAI,
Mistral, OpenRouter or a local Ollama also work) with tools from your MCP
servers (smart home, Nextcloud Deck, ...), and sends the outcome back to your
phone with ntfy. Risky actions wait for you to tap **Approve** (optionally, a
local Shieldstral classifier approves the plainly harmless ones for you; see
[Automatic approval](#automatic-approval-shieldstral)).

It is one Rust service in one container, built for a home server running
Docker Compose.

```
 ring ──BLE──► phone app ──HTTPS webhook──► Cloudflare Tunnel ──► ringmaster ─┬─► SQLite (/data)
                                                                              │
          ┌───────────────────────── worker ──────────────────────────────────┘
          │
          ├─► whisper-server on the Mac (LAN)   M4A → WAV in Rust, POST /inference
          │     └─ fallback Mac, then the phone's own transcription
          ├─► LLM (Claude, OpenAI, Mistral,     agent loop, tools as client tools
          │        OpenRouter, Ollama)
          │     └─► MCP servers (streamable HTTP, private compose network)
          └─► ntfy                              result / error / Approve-Deny
```

## How a recording is handled

1. **Webhook** (`POST /webhook`). The raw body is read first; the version
   header is checked, the HMAC signature verified in constant time with a
   5-minute timestamp window, the delivery id deduplicated, the multipart body
   parsed, the audio written to `/data/audio` and the job inserted in the same
   SQLite transaction. The app gets `202` in milliseconds; nothing here waits
   for the Mac or the LLM. A repeated delivery id gets `200` and is not
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
   the job, and runs a manual tool loop against the profile's LLM, executing
   tool calls itself so the MCP servers never leave the private network.
4. **Policies**. Every tool call is checked against `tool_policies`:
   `allow` runs, `deny` never runs (and is not even offered to the model),
   `confirm` pauses the job and sends an ntfy notification with Approve/Deny
   buttons. Tools that no rule matches need confirmation.
5. **Result**. The final answer, any error, and any delay are sent through
   ntfy. Everything is in the audit log.

Jobs are durable: a restart, a sleeping Mac, an API outage or a pending
confirmation all resume from the last saved step.

Nothing runs for ever: every job attempt, decode, model, MCP and ntfy call
has a deadline, idle MCP sessions are closed after 10 minutes, and shutdown
is bounded. See [Health and alerts](#health-and-alerts) for what is watched.

One job never holds up the others. Up to `agent.worker_concurrency` jobs (2)
run at once, each attempt has a deadline (`agent.attempt_timeout_secs`, 15
minutes) after which it is retried with back-off like an outage, a crash
while handling one job fails only that job, and a job pending Approve/Deny
does not occupy the worker. A queued job that can no longer be read (for
example a conversation saved by an incompatible version) is marked failed
(`job_quarantined` in the audit log) and you are notified with what you said,
instead of it blocking the queue.

## LLM providers

| `provider` | API | Key (env) | Default model |
|---|---|---|---|
| `anthropic` (default) | native Messages API, `https://api.anthropic.com/v1/messages` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `openai` | Chat Completions, `https://api.openai.com/v1/chat/completions` | `OPENAI_API_KEY` | none: set `model` |
| `mistral` | Chat Completions, `https://api.mistral.ai/v1/chat/completions` | `MISTRAL_API_KEY` | none: set `model` |
| `openrouter` | Chat Completions, `https://openrouter.ai/api/v1/chat/completions` | `OPENROUTER_API_KEY` | none: set `model` |
| `ollama` | native `/api/chat`, `http://localhost:11434` | none | none: set `model` |

The agent loop is the same for every provider: profiles, tool policies,
confirmations, the iteration cap, refusals, the "never run a cut-off tool
call" rule and the retry rules all behave identically. Each provider's
adapter stores the conversation in that provider's own message format and
echoes it back unchanged. A job keeps the provider and model it started
with, even if the config changes while it waits for a confirmation. Jobs
stored by older versions (no provider recorded) resume on Anthropic.

**Privacy.** With every provider except a local Ollama, the transcript of
what you said, the context (time, gesture), the tool definitions and every
tool result (device states, card titles, ...) are sent to that third party.
OpenRouter also passes them on to whichever upstream provider serves the
model. Keep a profile on Ollama if its data should stay on your network.

### Choosing a provider

Set the default in `[llm]`, and override it per profile with `provider` and
`model`. For example, capture on a local model and commands on Claude:

```toml
[llm]
provider = "anthropic"
model = "claude-opus-5-5"

[llm.ollama]
base_url = "http://m1-mac.lan:11434"

[profiles.capture]
provider = "ollama"
model = "qwen3:8b"
# ...
```

Keys directly under `[llm]` describe the default provider. A
`[llm.<provider>]` table configures one provider (base URL, key variable,
timeouts, default model) whether it is the default or only used by some
profiles, and wins over `[llm]`. A model must be given for every provider
except Anthropic: Ringmaster does not guess model names. `check-config`
fails on an unknown provider, a missing model, or an effort the provider
does not accept, and prints each profile's provider, model and endpoint
(with any password in the URL masked).

Only the providers that are actually used (the default, if any profile
relies on it, and every profile's `provider`) need their API key; the
service refuses to start without them. A key can come from `<NAME>_FILE` as
usual, and `api_key_env` names a different variable.

The legacy `[claude]` section still works unchanged: it is the Anthropic
configuration (`api_base` = `base_url`, and `model`, `effort`, `max_tokens`,
`max_iterations`, `refusal_fallbacks`, `request_timeout_secs`, `max_retries`
mean what they did). Anything set in `[llm]` or `[llm.anthropic]` takes
precedence over it.

### Claude (Anthropic)

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

### OpenAI, Mistral and OpenRouter

One OpenAI-style client with a preset per provider: `POST
<base_url>/v1/chat/completions` with `Authorization: Bearer <key>`, the
system prompt as the first message (`developer` role for OpenAI, `system`
for the others), MCP tools as `{"type": "function", ...}` definitions, and
each tool result as a `role: "tool"` message with the call's
`tool_call_id` (Mistral also gets the tool `name`). Failures and denials go
back as results starting with `Error:`.

* **Effort** is only sent when you set it: `reasoning_effort` for OpenAI and
  Mistral, `reasoning: {effort}` for OpenRouter. Which values a model
  accepts varies (OpenAI: `none` ... `max`; Mistral Small 4 only `none` or
  `high`). An unsupported value is a 400 from the provider.
* **Output limit**: `max_tokens` is only sent when set (as
  `max_completion_tokens` for OpenAI, which rejects `max_tokens` on reasoning
  models).
* **Refusals**: OpenAI's `message.refusal` and a `content_filter` finish are
  reported like Claude's refusals. Mistral guardrail blocks (HTTP 403) fail
  the job with the provider's message.
* **Cut-off replies**: `finish_reason: "length"` (or Mistral's
  `model_length`) fails the job without running any tool call in it.
* **Malformed tool calls**: arguments that are not a JSON object are never
  run. The model gets an error result asking it to call the tool again, so
  it can recover or explain.
* **Errors inside a 200** (OpenRouter reports some failed generations this
  way) are treated like the matching HTTP status. OpenAI quota and spend
  errors (`insufficient_quota`, `*_spend_limit_exceeded`, ...) are permanent
  even though they arrive as 429.
* **Reasoning round trips**: Mistral's thinking chunks and OpenRouter's
  `reasoning_details` are echoed back unchanged on the next turn.
* **Redirects are not followed**: a key header would go along to the new
  address. A `base_url` that redirects fails with an error naming the
  target; set `base_url` to that final address.
* **OpenRouter attribution**: set `[llm.openrouter] app_url` and `app_title`
  to send `HTTP-Referer` and `X-OpenRouter-Title`. Both are optional.
* **OpenAI models**: Ringmaster uses Chat Completions. Some current models
  (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`) only call tools through the
  Responses API, which Ringmaster does not speak. With those, the first
  request is rejected and the job runs without tools (see below). Pick a
  model whose page lists function calling on Chat Completions, e.g.
  `gpt-6.2-sol` or `gpt-5.4-mini`.

Example models (checked against the providers' model lists in November
2026): OpenAI `gpt-6.2-sol`, `gpt-5.4-mini`; Mistral `mistral-large-latest`,
`mistral-small-latest`; OpenRouter `anthropic/claude-sonnet-5.5`,
`openai/gpt-6.1-sol`, `mistralai/mistral-medium-3-5`.

### Ollama (local)

Ollama runs on a Mac on the LAN (an M1 with 16 GB is enough for an 8B
model). Ringmaster uses Ollama's native `POST /api/chat`, not its
OpenAI-compatible endpoint, because only the native API can set the context
window per request. Ollama's own default is 4096 tokens on a 16 GB machine,
and it silently drops the oldest messages when a conversation outgrows it.
Ringmaster sends `options.num_ctx` (default 8192; `num_ctx = 0` leaves it to
the server) and `truncate: false`, so an overflow is an error instead of a
quietly forgotten transcript.

* No API key. If Ollama sits behind a reverse proxy that wants one, set
  `[llm.ollama] api_key_env` and it is sent as a bearer token.
* Make Ollama listen on the LAN: on the Mac,
  `launchctl setenv OLLAMA_HOST 0.0.0.0:11434` (or set it in the Ollama
  app's settings) and restart Ollama. Then set
  `[llm.ollama] base_url = "http://<mac>.lan:11434"`.
* Keep the model loaded between requests, or the first one waits for it to
  load: `launchctl setenv OLLAMA_KEEP_ALIVE 30m` (or `-1`).
* Local models are slow: the default timeout is 600 s with one retry
  (`request_timeout_secs`, `max_retries`). Raise `agent.job_max_age_secs` (or
  the profile's) if the Mac may be busy for longer, and
  `agent.attempt_timeout_secs` (900 s) if one job can legitimately take
  longer than that.
* Use a model with tool support, such as `qwen3:8b` (thinking and tools),
  `llama3.1:8b` or `qwen2.5:7b`. A model without tools rejects the request
  (`does not support tools`); the job then runs once more without tools and
  the reply ends with `Unavailable: all tools`, so nothing hangs.
* `effort` maps to Ollama's `think`: `none` turns thinking off, `low`,
  `medium`, `high` or `max` set its level. Unset leaves the model default.
* Small models sometimes produce tool calls Ollama cannot parse. Ollama then
  answers 500 (`failed to parse JSON`); Ringmaster fails that job with a
  clear message instead of retrying it for an hour. An empty first reply
  (no text, no tool call) also fails the job rather than claiming "Done."

### The tool fallback, for every provider

If the API rejects the very first request of a job with a 400 while tools
are offered, the request is retried once without tools (unless the error is
about thinking or reasoning, which dropping tools would not fix). The model
can still answer and say what it could not do, and the reply ends with
`Unavailable: all tools`.

## Configuration

`config.toml` (path from `--config` or `RINGMASTER_CONFIG`, default
`/etc/ringmaster/config.toml`). See [`config.example.toml`](config.example.toml)
for a complete, commented example. `ringmaster check-config` validates a file.

### Secrets (environment only)

Each can also be given as `<NAME>_FILE` pointing at a file.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (required when a profile uses `anthropic`, the default) |
| `OPENAI_API_KEY` | OpenAI API key (when a profile uses `openai`) |
| `MISTRAL_API_KEY` | Mistral API key (when a profile uses `mistral`) |
| `OPENROUTER_API_KEY` | OpenRouter API key (when a profile uses `openrouter`) |
| any `api_key_env` named in `[llm]` / `[llm.<provider>]` | replaces the default variable for that provider |
| `RINGMASTER_WEBHOOK_SECRET` | the ring app's signing secret, at least 32 characters (required unless `webhook.require_signature = false`) |
| `NTFY_TOKEN` | ntfy access token, if the topic needs one |
| the `approval.api_key_env` variable, if set | bearer token for the approval classifier server (a missing one is a warning; calls then go to you) |
| any `token_env` named by an `[[mcp_servers]]` entry | bearer token for that MCP server |

### Reference

| Key | Default | Meaning |
|---|---|---|
| `server.listen` | `0.0.0.0:8080` | HTTP listen address |
| `server.public_base_url` | none | external URL (tunnel hostname) for the Approve/Deny links; without it confirmations have no buttons |
| `server.data_dir` | `/data` | SQLite database and recordings |
| `server.max_body_bytes` | 25 MiB | webhook body limit |
| `server.shutdown_grace_secs` | `8` | on SIGTERM, how long in-flight requests and jobs get to finish; keep it below the container's stop timeout (Docker: 10 s, or `stop_grace_period`) |
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
| `stt.max_audio_bytes` / `stt.max_audio_secs` | 25 MiB / `600` | larger or longer recordings are not decoded; the phone's transcription is used |
| `stt.decode_timeout_secs` | `60` | wall-clock limit for decoding one recording (checked inside the decode loop) |
| `stt.decode_concurrency` | `1` | decoder threads at once (1-8); also the most threads a hung decoder can tie up |
| `llm.provider` | `anthropic` | default provider: `anthropic` `openai` `mistral` `openrouter` `ollama` |
| `llm.model` | `claude-opus-5-5` for Anthropic, otherwise required | default provider's model |
| `llm.effort` | `low` for Anthropic, otherwise unset | Anthropic: `low` `medium` `high` `xhigh` `max`; OpenAI/OpenRouter: `none` `minimal` `low` `medium` `high` `xhigh` `max`; Mistral: `none` `high`; Ollama: `none` `low` `medium` `high` `max` |
| `llm.base_url` / `llm.api_key_env` | per provider (table above) | default provider's endpoint (without `/v1`) and key variable |
| `llm.request_timeout_secs` / `llm.max_retries` | `120` / `4`; Ollama `600` / `1` | default provider's per-request timeout and retries |
| `llm.max_tokens` | `16000` for Anthropic, otherwise unset (provider default) | output limit, thinking included |
| `llm.max_iterations` | `8` | model calls per job, any provider |
| `[llm.<provider>]` `model`, `effort`, `base_url`, `api_key_env`, `request_timeout_secs`, `max_retries`, `max_tokens` | | the same settings for one provider; win over `[llm]` |
| `llm.anthropic.refusal_fallbacks` | `auto` | `auto` `on` `off` |
| `llm.openrouter.app_url` / `app_title` | none | OpenRouter attribution headers |
| `llm.ollama.num_ctx` | `8192` | context window asked of Ollama; `0` = server default |
| `claude.*` | as before | legacy Anthropic settings (`api_base`, `model`, `effort`, `max_tokens`, `max_iterations`, `refusal_fallbacks`, `request_timeout_secs`, `max_retries`); `[llm]` wins |
| `agent.timezone` | `UTC` | IANA zone for the time given to the model |
| `agent.base_system_prompt` | built in | replaces the shared preamble |
| `agent.confirmation_ttl_secs` | `600` | Approve/Deny link lifetime; no answer counts as deny |
| `agent.job_max_age_secs` | `3600` | give up (and say so) after this; profiles override |
| `agent.retry_base_secs` / `agent.retry_max_secs` | `15` / `600` | job back-off |
| `agent.notify_on_retry` | `true` | one "delayed" notification per job |
| `agent.tool_timeout_secs` | `60` | per MCP call |
| `agent.tool_discovery_timeout_secs` | `10` | listing a server's tools; slower servers are skipped for that job |
| `agent.max_tool_result_chars` | `20000` | tool output truncation |
| `agent.attempt_timeout_secs` | `900` | wall-clock limit for one attempt at a job; an attempt that runs over is retried with back-off until the job's max age |
| `agent.worker_concurrency` | `2` | jobs processed at once (1-16), so one slow or stuck job does not hold up the queue; with more than 1, jobs can finish out of order |
| `retention.audio_days` | `30` | delete recordings after this (0 keeps them) |
| `alerts.enabled` | `true` | send operational alerts (they are always logged and on `/healthz`, `/metrics`) |
| `alerts.ntfy_topic` | `notify.ntfy.topic` | separate ntfy topic for alerts (same server and token) |
| `alerts.check_interval_secs` | `30` | how often the watchdog checks |
| `alerts.reminder_secs` | `3600` | repeat an alert this often while it lasts (0 = never) |
| `alerts.db_check_interval_secs` | `300` | `PRAGMA quick_check` and a write probe; also run at once after a database error |
| `alerts.db_error_threshold` | `3` | statements (or job claims) failing in a row before an alert; corruption, read-only, full and I/O errors alert at once |
| `alerts.worker_stall_secs` | `120` | the worker loop has not run for this long (it runs at least every 30 s; minimum 60) |
| `alerts.queue_stall_secs` | `600` | a due job has waited this long to start |
| `alerts.overrun_margin_secs` | `60` | a job (or a decode) this far past its deadline |
| `alerts.failure_threshold` / `alerts.failure_window_secs` | `3` / `900` | this many jobs failed, expired, quarantined or panicked within the window |
| `notify.ntfy.url` / `.topic` | `https://ntfy.sh` / required | |
| `notify.ntfy.action_headers` | `{}` | headers the phone sends with Approve/Deny |
| `[[mcp_servers]]` `name`, `url`, `token_env` | | streamable HTTP MCP servers; `token_env` sends `Authorization: Bearer` |
| `[[mcp_servers]]` `basic_auth` | | `{ username = "...", password_env = "VAR" }`: HTTP Basic, e.g. a Nextcloud app password; takes precedence over `token_env` |
| `[[mcp_servers]]` `headers_env` | | `{ "X-Api-Key" = "VAR" }`: extra headers, values from environment variables |
| `[[tool_policies]]` `match`, `policy` | | glob over `server/tool`; `allow` `confirm` `ask` `deny`; first match wins (`ask` = `confirm` that the classifier may never approve) |
| `[profiles.<name>]` `triggers`, `system_prompt`, `tools` | | gesture → prompt + tool allow-list (globs; `*/*` = all) |
| `[profiles.<name>]` `provider`, `model`, `effort`, `max_iterations`, `job_max_age_secs` | | per-profile overrides; a profile with a `provider` other than the default needs a `model` (here or in `[llm.<provider>]`) |
| `[profiles.<name>]` `approval_mode` | `approval.mode` | `manual` `shadow` `classifier` for this profile |
| `approval.mode` | `manual` | who approves `confirm` calls: `manual` (you), `shadow` (you, with the classifier's verdict recorded), `classifier` (the classifier for clear passes, you for the rest); see [Automatic approval](#automatic-approval-shieldstral) |
| `approval.base_url` | none; required unless every mode is `manual` | classifier server (llama-server, vLLM, SGLang), without `/v1` |
| `approval.model` | `mistralai/Shieldstral-1.0-3B` | model name in the request (vLLM checks it, llama-server ignores it) |
| `approval.api_key_env` | none | variable holding a bearer token for the classifier server |
| `approval.request_timeout_secs` / `approval.max_retries` | `15` / `1` | how long to wait for a verdict, retries included (1-120), then ask you |
| `approval.threshold` | `0.5` | scores above this are reported as "flagged" |
| `approval.auto_approve_below` | `approval.threshold` | only scores below this run without asking; between it and `threshold` is a grey band that asks you (0.2 recommended) |
| `approval.instruct` / `approval.query` | built in | the `<Instruct>` and `<Query>` text; phrase the query so "yes" means unsafe |
| `approval.alert_after_failures` | `3` | classifier failures in a row before an alert |

Glob `*` does not cross the `/` between server and tool: `homey/*` matches
every Homey tool, `*` alone matches nothing.

MCP problems never fail a job or stop the service. For each job, every MCP
server is listed concurrently with a short timeout
(`agent.tool_discovery_timeout_secs`, default 10). A server that is down, slow
or returns an error is skipped, and so is any tool whose input schema the
model APIs would reject. If the API still rejects the tool list, the request
is retried once with no tools. The model is told which servers are unavailable,
the reply ends with `Unavailable: ... You said: "..."` so the request is never
lost, and the skip is recorded in the audit log (`tool_server_skipped`,
`tool_skipped`, `tools_dropped`). A tool call that fails or times out goes
back to the model as an error result. A malformed `[[mcp_servers]]` entry is
ignored with a warning at startup; `check-config` reports it as an error.

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
| result | 3 | the agent finished; the body is the model's one-line answer |
| approve? | 5 | a `confirm` or `ask` tool is waiting; `Approve` / `Deny` buttons (`http` actions that `POST` to `/confirm/<token>/approve|deny`) and a `Details` link; with an approval classifier, a line per call with its verdict and score, and the title `Ringmaster: approve? (flagged by the classifier)` when it flagged one |
| problem | 4 | errors, refusals, give-ups, and the first delay of a job |
| info | 2 | test events, empty recordings |
| alert | 5 (reminders 4) | operational problems (see [Health and alerts](#health-and-alerts)); tags `rotating_light`, `ringmaster-alert`; title `Ringmaster alert: ...` |
| recovered | 3 | an alerted problem has cleared; tags `heavy_check_mark`, `ringmaster-alert` |

Set `alerts.ntfy_topic` to send alerts to their own topic.

## Automatic approval (Shieldstral)

By default every `confirm` tool call waits for you to tap Approve. Optionally,
a local [Shieldstral](https://huggingface.co/mistralai/Shieldstral-1.0-3B)
classifier (Mistral's 3.8B safety classifier, Apache 2.0) can approve the
plainly harmless ones, so "turn on the kitchen lights" does not need a tap.
It runs on your own hardware behind an OpenAI-compatible endpoint; see
[docs/mac-shieldstral.md](docs/mac-shieldstral.md) for llama.cpp on a Mac.

> **Read this first.** Shieldstral was trained and evaluated on *content*
> safety: harmful prompts, responses and images. It was **not** trained or
> evaluated on judging an agent's tool calls. Ringmaster shows it your words
> and the proposed call as a document and asks whether the action goes beyond
> what you asked for or could cause harm; how well it answers that is
> untested. Small classifiers are particularly weak at what matters most
> here: noticing that speech was misheard, or that the action targets the
> wrong door, card or person when it looks harmless on its own. The
> transcript is also untrusted text that the classifier reads, so a crafted
> request could try to talk it into "no". Treat it as a convenience for
> low-stakes actions, never as a safety boundary: profiles and policies stay
> that boundary.

How each mode behaves (`approval.mode`, or `approval_mode` per profile):

| Mode | `confirm` calls | `ask` calls | `allow` / `deny` |
|---|---|---|---|
| `manual` (default) | Approve/Deny notification, exactly as before | same as `confirm` | unchanged |
| `shadow` | Approve/Deny notification; the classifier's verdict is shown in it and recorded, and compared with your decision | Approve/Deny, never classified | unchanged |
| `classifier` | score below `auto_approve_below`: runs at once; anything else (flagged, grey band, no verdict): Approve/Deny | Approve/Deny, never classified | unchanged |

* The classifier **can approve but never deny.** A "flagged" verdict only
  means you are asked, with the classifier's score in the notification
  ("Classifier flagged this (score 0.93 > 0.50)").
* **It fails over to you.** A server that is down, slow
  (`request_timeout_secs`), answers with an error or without `yes`/`no`
  log-probabilities, or a call too long to judge whole (transcript over 2000
  or arguments over 4000 characters) means you are asked, never that the
  call runs or the job fails. After `alert_after_failures` failures in a row
  you get an operational alert.
* `deny` tools are never offered to the model and never classified; `ask`
  tools always ask you and are never sent to the classifier. If one call in
  a turn is `ask`, the others are not classified either.
* Each call in a turn is classified on its own; the turn runs without asking
  only if every `confirm` call in it passes. Otherwise the whole turn goes to
  you, as without a classifier, with a line per call saying what the
  classifier made of it.
* Every auto-approved action is listed, with its arguments and score, at the
  end of the job's final notification ("Auto-approved by the classifier:
  home/set_light {...} (score 0.03)"), and in the audit log.

The request follows the model card: the fixed system message, a user message
of `<Instruct>` (the context and strictness), `<Query>` (one yes/no question
where "yes" means "do not run without asking") and `<Document>` (what you
said, verbatim; any text the model wrote with the call; the tool's name,
description and exact arguments), with `max_tokens = 1`, `temperature = 0`,
`logprobs = true` and `top_logprobs = 20`. The score is the softmax of the
`yes` and `no` log-probabilities. Override the wording with
`approval.instruct` and `approval.query`.

**Recommended rollout**

1. Run `shadow` for a few weeks. Every `confirm` call still asks you; the
   notification shows what the classifier would have done, and each decision
   is recorded as a `shadow_comparison` audit row
   (`ringmaster_shadow_comparisons_total{result}`). Look hard at
   `classifier_would_have_run_a_denied_action`: each one is a call it would
   have run that you refused.
2. Set deletes and anything destructive to `deny` (they are then never even
   offered), and locks, alarms, doors, payments and messages to other people
   to `ask`.
3. Switch to `classifier` with a grey band (`auto_approve_below = 0.2`), and
   keep `manual` on profiles where a wrong action is costly
   (`approval_mode = "manual"`). Watch the auto-approved lists in your
   notifications.

```toml
[approval]
mode = "classifier"
base_url = "http://192.168.88.30:8090"   # llama-server on the Mac
auto_approve_below = 0.2

[[tool_policies]]
match = "*/*delete*"
policy = "deny"

[[tool_policies]]
match = "homey/*lock*"
policy = "ask"

[[tool_policies]]
match = "homey/*alarm*"
policy = "ask"
```

## Transcription on the Mac

For the optional approval classifier, see
[docs/mac-shieldstral.md](docs/mac-shieldstral.md) (llama.cpp `llama-server`).

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
| `GET /healthz` | `200 ok`, or `503 unhealthy: <reasons>` (see [Health and alerts](#health-and-alerts)) |
| `GET /metrics` | Prometheus metrics |

`ringmaster healthcheck` probes `/healthz` for the Docker `HEALTHCHECK`
(the image has no shell or curl).

### Logs and metrics

Logs are JSON on stdout (`RUST_LOG` sets levels; `RINGMASTER_LOG_FORMAT=text`
for humans). Metrics include `ringmaster_webhook_requests_total{result}`,
`ringmaster_jobs_total{outcome}`, `ringmaster_queue_depth`,
`ringmaster_stt_requests_total{source,result}`,
`ringmaster_llm_requests_total{provider,status}`,
`ringmaster_llm_request_seconds{provider}`,
`ringmaster_llm_tokens_total{provider,kind}` (input, output, cache_read,
cache_write; up to 0.1.2 these were `ringmaster_claude_*` without a provider
label), `ringmaster_tool_calls_total{outcome}`,
`ringmaster_confirmations_total{decision}`,
`ringmaster_notifications_total{status}`, and for the approval classifier
`ringmaster_approvals_total{mode,outcome}` (per turn needing approval:
`auto_approved`, `escalated`, `shadow`, `manual`),
`ringmaster_classifications_total{mode,verdict}` (`safe`, `uncertain`,
`unsafe`, `unavailable`, `ask`, `skipped`),
`ringmaster_classifier_score{mode}` (histogram),
`ringmaster_shadow_comparisons_total{result}` and the classifier's requests
as `ringmaster_llm_requests_total{provider="classifier",status}`.

Health gauges (refreshed by the watchdog and on every scrape):
`ringmaster_db_healthy` (0/1), `ringmaster_db_errors_total{class}`,
`ringmaster_jobs_running`, `ringmaster_jobs_due`,
`ringmaster_oldest_due_job_age_seconds`,
`ringmaster_worker_heartbeat_age_seconds`,
`ringmaster_watchdog_heartbeat_age_seconds`, `ringmaster_alerts_active`,
`ringmaster_alert_active{condition}`,
`ringmaster_classifier_failures_in_a_row`, `ringmaster_blocking_tasks{kind}`
(decodes and audio writes in progress), `ringmaster_mcp_sessions` and
`ringmaster_tokio_alive_tasks`. A steady rise in the last three is a leak.

### Health and alerts

A watchdog task, independent of the worker, checks every
`alerts.check_interval_secs` (30 s) and alerts you through the notifier:

| Alert | When |
|---|---|
| database corrupt | a statement fails with `SQLITE_CORRUPT`/`SQLITE_NOTADB`, or `PRAGMA quick_check` (on a fresh connection, every `db_check_interval_secs`) finds problems |
| database failing | writes fail with `SQLITE_FULL`, `SQLITE_READONLY` or an I/O error (a write probe confirms; alerts at once), `db_error_threshold` statements or job claims fail in a row, or the database stops answering |
| worker stalled | the worker loop has not run for `worker_stall_secs` |
| queue stalled | a due job has not started for `queue_stall_secs` |
| job past its deadline | a job has run `overrun_margin_secs` past `agent.attempt_timeout_secs` (the deadline did not fire) |
| jobs failing | `failure_threshold` jobs failed, expired, were quarantined or panicked within `failure_window_secs` |
| blocking work stuck | a decode (or an audio write) has not returned `overrun_margin_secs` after its limit |
| approval classifier failing | `approval.alert_after_failures` classifications in a row got no verdict (the calls went to you); never affects `/healthz` |

Each problem gets one alert when it starts, a reminder every `reminder_secs`
while it lasts and one "recovered" message when it clears. Alerts carry
counts, job ids, ages and error classes, never what you said or a secret.
Example:

> **Ringmaster alert: database failing**
> Database writes fail: disk is full (SQLITE_FULL). New recordings may not
> be stored or processed; the webhook answers 503 so the phone retries (up
> to 10 times, a minute apart).

Alerts are always logged. If ntfy is down the alert is logged and retried
a minute later; sending never blocks or stops anything. The alert state is
kept in `/data/alerts.json`, so a restart that fixes a problem still sends
"recovered", and a restart does not repeat an alert.

While the database is corrupt the webhook answers `503` before storing
anything (as it already did for any storage error), so the phone keeps the
recording and retries rather than getting a `2xx` for data that was not
safely stored. If the database cannot be opened at start-up, Ringmaster
alerts and retries every 30 s instead of exiting, so the container does not
restart in a loop.

`/healthz` returns `503` with the reasons only when recordings cannot be
stored or processed at all: the database is corrupt, read-only, full,
failing I/O or not answering (a 2 s `SELECT 1`), or the worker loop or the
watchdog has stopped. It reads in-memory state and never runs the
expensive checks itself. Slow or failing jobs, a stalled queue, a sleeping
Mac or a down MCP server are alerted on but keep it `200`: restarting does
not fix them. Docker's `HEALTHCHECK` only marks the container unhealthy;
Compose's `restart: unless-stopped` acts on exits, not health, so an
unhealthy Ringmaster is never restarted in a loop. If you add an autoheal
tool, a restart is safe: shutdown is graceful, jobs resume, and webhooks
refused meanwhile are retried by the phone.

### Shutdown

On SIGTERM the listener closes (new webhooks are refused and retried by the
phone), in-flight requests and jobs get `server.shutdown_grace_secs` to
finish, the watchdog and housekeeping stop, and the process exits. A job
still running then stays `running` and is requeued at the next start. Set
the container's stop timeout above the grace period
(`stop_grace_period: 30s` with `shutdown_grace_secs = 25` in the compose
example).

### Audit log

`/data/ringmaster.db` (SQLite, WAL). `deliveries` has every webhook delivery;
`jobs` the queue with transcript, transcript source and the stored
conversation; `confirmations` the hashed one-time tokens and decisions;
`audit` a timeline per job: `delivery`, `transcript`, `agent_start`,
`claude_response` (stop reason, tool calls, usage; `llm_response` with the
provider and finish reason for the other providers), `tool_call` (input,
policy action, result, `approved_by` `human` or `classifier`),
`classification` (per classified call: tool, mode, verdict, score,
`threshold`, `auto_approve_below`, outcome `auto_approved` / `escalated` /
`shadow`, error), `confirmation_requested` (with the classifier's
judgements) / `_answered` / `_decision`, `shadow_comparison` (your decision
against what the classifier would have done),
`retry_scheduled`, `job_finished` / `job_failed`, and `job_quarantined` for
a queued job that could no longer be read.

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
* MCP tool calls are at-least-once: if Ringmaster dies (or an attempt hits
  `attempt_timeout_secs`) after a tool ran but before the result was saved,
  the call runs again on the next attempt.
* A job that was already started is not resumed after a restart if it is
  more than `job_max_age_secs` + `retry_max_secs` + `confirmation_ttl_secs`
  old: a long outage, or a job that keeps crashing the service, does not
  carry out a stale half-done request.
* With `worker_concurrency` above 1, two commands spoken close together may
  run at the same time and finish in either order. Set it to 1 for strict
  order; a stuck job then delays the rest by up to `attempt_timeout_secs`.
* The transcript is untrusted input. The capture profile only has Deck
  tools, so even "unlock the door" said in capture mode becomes a card.
* The approval classifier is a convenience, not a safety boundary: it was
  not trained on tool calls and can be wrong or talked round. Keep
  destructive tools on `deny` and security-relevant ones on `ask`, and start
  in `shadow` mode (see [Automatic approval](#automatic-approval-shieldstral)).

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
(`tests/fixtures/tone-1s.m4a`), the STT fallback chain, LLM retries, and
the whole pipeline end to end against mocked model APIs (wiremock: Claude,
OpenAI, Mistral, OpenRouter and Ollama request and message shapes) and a
real MCP server (rmcp) on localhost, including approve and deny.

## Releases

Conventional Commits drive [release-please](https://github.com/googleapis/release-please):
it keeps a release PR open; merging it tags `vX.Y.Z`, publishes the GitHub
release with notes, and pushes `ghcr.io/lazzurs/ringmaster:X.Y.Z`, `X.Y` and
`latest` for linux/amd64 and linux/arm64 with SBOM and provenance
attestations. See `.github/workflows/release.yml` for how the release PR gets
its required checks.

## License

Copyright (C) 2026 Rob Lazzurs

Ringmaster is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See [`LICENSE`](LICENSE) for the full text.

Not affiliated with or endorsed by the maker of the ring or its app.
