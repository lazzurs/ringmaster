# Shieldstral on the Mac (llama.cpp)

Ringmaster's optional automatic approval (`[approval]`, see the README
section "Automatic approval (Shieldstral)") asks a local
[Shieldstral](https://huggingface.co/mistralai/Shieldstral-1.0-3B) classifier
whether a proposed tool call is safe to run without asking you. This page
runs it with llama.cpp's `llama-server` on an Apple Silicon Mac (Metal), for
example a spare 16 GB M1.

Checked against llama.cpp **v1.0.0** (`tools/server/server-common.cpp`,
`server-task.cpp`, `conversion/base.py`) and the Shieldstral 1.0 model card
as of November 2026. Nothing here has been run against a real Shieldstral
by the Ringmaster tests: they use a mock server that answers in
llama-server's format.

## What Ringmaster needs from the server

One `POST /v1/chat/completions` per proposed tool call, as the model card
describes:

```json
{
  "model": "mistralai/Shieldstral-1.0-3B",
  "messages": [
    {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruction provided. Note that the answer can only be \"yes\" or \"no\"."},
    {"role": "user", "content": "<Instruct>: ...\n\n<Query>: ...\n\n<Document>: [User request]\n...\n\n[Proposed action]\nTool: ...\nDescription: ...\nArguments: {...}"}
  ],
  "max_tokens": 1, "temperature": 0.0, "logprobs": true, "top_logprobs": 20, "stream": false
}
```

and, in the reply, `choices[0].logprobs.content[0].top_logprobs`: a list of
`{"token", "logprob"}`. llama-server supports this: `logprobs: true` turns on
its `n_probs` with `top_logprobs` alternatives (20 by default), and the
non-streaming reply carries `choices[0].logprobs.content[]` with `token`,
`logprob`, `bytes` and `top_logprobs` per generated token. The score is
`exp(yes) / (exp(yes) + exp(no))` over the best `yes` and `no` spellings
(`yes`, `Yes`, `yes.`, `"yes"`, ... after trimming and lowercasing), each
clamped at -10 as in the model card's helper. A reply with neither a `yes`
nor a `no` among the alternatives is treated as no verdict, and the call
goes to you.

## 1. Install llama.cpp

`llama-server` from Homebrew is built with Metal:

```sh
brew install llama.cpp
llama-server --version            # 1.0.0 or later
```

The GGUF conversion script is not part of the formula; clone the matching
source for it:

```sh
mkdir -p ~/src && cd ~/src
git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp
cd llama.cpp
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements/requirements-convert_hf_to_gguf.txt
pip install "mistral-common>=1.11.5" "huggingface_hub[cli]"
```

## 2. Get a GGUF

Convert Mistral's own weights (Apache 2.0, not gated at the time of
writing). Only the Mistral-format files are needed:

```sh
cd ~/src/llama.cpp
hf download mistralai/Shieldstral-1.0-3B \
    --exclude "model.safetensors" \
    --local-dir ~/models/Shieldstral-1.0-3B

python convert_hf_to_gguf.py ~/models/Shieldstral-1.0-3B \
    --mistral-format --outtype bf16 \
    --outfile ~/models/Shieldstral-1.0-3B-BF16.gguf

# Optional, smaller and faster: 8-bit.
llama-quantize ~/models/Shieldstral-1.0-3B-BF16.gguf \
    ~/models/Shieldstral-1.0-3B-Q8_0.gguf Q8_0
```

The checkpoint ships a `chat_template.jinja`, and the converter embeds it in
the GGUF (it logs "Using an existing Mistral local chat template"). The
vision projector (`--mmproj`) is not needed: Ringmaster only sends text.

Community GGUFs exist on Hugging Face (search "Shieldstral GGUF"), but they
are third-party builds of a safety model; converting the official weights
yourself is the safer choice.

**Memory.** The model is about 3.8B parameters: roughly 7.5 GB in BF16 and
about 4 GB in Q8_0, plus the context. On a 16 GB M1, macOS lets the GPU wire
about two-thirds to three-quarters of RAM, so BF16 fits but leaves little for
anything else (do not also run whisper `large-v3` on the same Mac with BF16);
Q8_0 is the comfortable choice. Quantising can move scores a little against
the published (BF16) results: another reason to run in `shadow` mode first.
`Q5_K_M` and `Q4_K_M` save more memory and lose more accuracy.

## 3. Try it by hand

```sh
llama-server \
    -m ~/models/Shieldstral-1.0-3B-Q8_0.gguf \
    --host 192.168.88.30 --port 8090 \
    -c 16384 -np 2 \
    --jinja --no-webui
```

* `-c 16384 -np 2`: two slots of 8192 tokens each. A Ringmaster request is a
  few thousand tokens at most (the transcript is capped at 2000 characters
  and the arguments at 4000; longer calls are never sent and go to you), and
  two slots let two calls of one turn, or two jobs, be judged at once.
* `--jinja` (the default in current builds) uses the model's own chat
  template. Without a correct template the classifier would see the wrong
  prompt format and its scores would mean little.
* `--no-webui` turns off the built-in chat page.
* `--host`: the Mac's LAN address, not `0.0.0.0`. llama-server has no
  authentication unless you give it `--api-key`; if you do, put the same key
  in an environment variable on the Ringmaster side and name it in
  `approval.api_key_env`.

From the homeserver, check that the logprobs come back:

```sh
curl -s http://192.168.88.30:8090/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruction provided. Note that the answer can only be \"yes\" or \"no\"."},
    {"role": "user", "content": "<Instruct>: You are a strict safety reviewer.\n\n<Query>: Does this proposed action go beyond what the user asked for?\n\n<Document>: [User request]\nturn on the kitchen lights\n\n[Proposed action]\nTool: homey/set_light\nArguments: {\"name\":\"kitchen\",\"on\":true}"}
  ],
  "max_tokens": 1, "temperature": 0, "logprobs": true, "top_logprobs": 20
}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print([(t["token"], round(t["logprob"],3)) for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"][:5]])'
# illustrative: [('no', -0.01), ('yes', -4.6), ...]  -> score about 0.01
```

If `logprobs` is missing from the reply, the server is too old: Ringmaster
will then never get a verdict and asks you every time (and alerts after
`approval.alert_after_failures` failures in a row).

The first request after start-up is slower (Metal compiles its kernels).
After that the time is mostly prompt processing of roughly 500 to 2000
tokens, which should be a few seconds at most on an M1; this has not been
measured for Ringmaster, so time a few requests yourself. Ringmaster waits
`approval.request_timeout_secs` (default 15) and then asks you instead.

## 4. Run it at login

Same pattern as whisper-server in
[mac-transcription.md](mac-transcription.md) (LaunchAgent, automatic login,
`pmset` so the Mac never sleeps). Save as
`~/Library/LaunchAgents/ie.lazzurs.shieldstral.plist` (absolute paths;
replace `rob` and the address; Homebrew on Apple Silicon installs to
`/opt/homebrew/bin`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ie.lazzurs.shieldstral</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/llama-server</string>
    <string>-m</string>
    <string>/Users/rob/models/Shieldstral-1.0-3B-Q8_0.gguf</string>
    <string>--host</string>
    <string>192.168.88.30</string>
    <string>--port</string>
    <string>8090</string>
    <string>-c</string>
    <string>16384</string>
    <string>-np</string>
    <string>2</string>
    <string>--jinja</string>
    <string>--no-webui</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>/Users/rob/Library/Logs/shieldstral.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/rob/Library/Logs/shieldstral.log</string>
</dict>
</plist>
```

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ie.lazzurs.shieldstral.plist
launchctl print gui/$(id -u)/ie.lazzurs.shieldstral | grep -E 'state|pid'
curl -s http://192.168.88.30:8090/health     # {"status":"ok"} once loaded
```

After a `brew upgrade llama.cpp`: `launchctl kickstart -k
gui/$(id -u)/ie.lazzurs.shieldstral`, and run the curl check above again.

## 5. Point Ringmaster at it

```toml
[approval]
mode = "shadow"                       # watch it for a while first
base_url = "http://192.168.88.30:8090"
auto_approve_below = 0.2
```

Then follow the rollout in the README: compare its verdicts with your own
decisions (`classification` and `shadow_comparison` audit rows,
`ringmaster_shadow_comparisons_total`), and only then switch to
`mode = "classifier"` with deletes on `deny` and locks and alarms on `ask`.
