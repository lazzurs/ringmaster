# Transcription on the Mac (whisper.cpp)

Ringmaster does not transcribe audio itself. It converts each recording to a
16 kHz mono WAV and posts it to whisper.cpp's `whisper-server`, running
natively on an Apple Silicon Mac so it can use the GPU through Metal (Docker
on macOS cannot). This page sets that up on an M1 so it survives reboots,
power cuts and nobody being logged in.

Checked against whisper.cpp **v1.9.6** (`examples/server/server.cpp`).

## 1. Build whisper-server

The Homebrew `whisper-cpp` formula builds with `-DWHISPER_BUILD_SERVER=OFF`,
so it ships `whisper-cli` but **not** `whisper-server`. Build from source
instead (Metal is on by default on macOS):

```sh
xcode-select --install            # once, for the compilers
brew install cmake git

mkdir -p ~/src && cd ~/src
git clone --depth 1 --branch v1.9.6 https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_SERVER=ON
cmake --build build -j --config Release

ls build/bin/whisper-server       # the binary the LaunchAgent runs
```

## 2. Download the model

`large-v3-turbo` is the best accuracy/speed trade-off for short commands on an
M1 (about 1.6 GB; `large-v3-turbo-q5_0` is a smaller quantised variant):

```sh
cd ~/src/whisper.cpp
./models/download-ggml-model.sh large-v3-turbo
ls -lh models/ggml-large-v3-turbo.bin
```

## 3. Try it by hand

```sh
./build/bin/whisper-server \
  --model models/ggml-large-v3-turbo.bin \
  --host 0.0.0.0 --port 8178 \
  --language en --threads 4
```

From the homeserver:

```sh
curl -s http://mac-mini.lan:8178/health
# {"status":"ok"}   (503 {"status":"loading model"} while it starts)

curl -s http://mac-mini.lan:8178/inference \
  -F file=@samples/jfk.wav -F response_format=json -F temperature=0.0
# {"text":" And so my fellow Americans, ..."}
```

Notes on the flags:

* `--host 0.0.0.0` listens on every interface. Prefer the Mac's LAN address
  (e.g. `--host 192.168.88.20`) so it is not reachable on other interfaces.
  `whisper-server` has **no authentication**: keep it on the LAN, and allow
  it through the macOS firewall only from the homeserver if you can.
* Flash attention is on by default in this version (`-nfa` turns it off).
* `--threads 4` matches the M1's four performance cores; Metal does the heavy
  lifting either way.
* Do **not** pass `--convert`: it needs ffmpeg on the Mac. Ringmaster sends
  WAV already.
* Ringmaster sends `language`, `temperature`, `response_format=json` and an
  initial `prompt` built from `[stt] vocabulary` with every request, so the
  server's own defaults for those do not matter.

Then set `[stt] primary_url = "http://mac-mini.lan:8178"` in Ringmaster's
config. A second Mac can be `fallback_url`.

## 4. Run it at login with launchd

Save as `~/Library/LaunchAgents/ie.lazzurs.whisper-server.plist` (paths must
be absolute; replace `rob` and the address):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ie.lazzurs.whisper-server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/rob/src/whisper.cpp/build/bin/whisper-server</string>
    <string>--model</string>
    <string>/Users/rob/src/whisper.cpp/models/ggml-large-v3-turbo.bin</string>
    <string>--host</string>
    <string>192.168.88.20</string>
    <string>--port</string>
    <string>8178</string>
    <string>--language</string>
    <string>en</string>
    <string>--threads</string>
    <string>4</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/rob/src/whisper.cpp</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <!-- Do not let macOS throttle it as a background job. -->
  <key>ProcessType</key>
  <string>Interactive</string>
  <!-- Back off a little if it crash-loops (e.g. bad model path). -->
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>/Users/rob/Library/Logs/whisper-server.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/rob/Library/Logs/whisper-server.log</string>
</dict>
</plist>
```

Load it and check:

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ie.lazzurs.whisper-server.plist
launchctl print gui/$(id -u)/ie.lazzurs.whisper-server | grep -E 'state|pid'
tail -f ~/Library/Logs/whisper-server.log
```

After editing the plist: `launchctl bootout gui/$(id -u)/ie.lazzurs.whisper-server`
and bootstrap again. After a whisper.cpp upgrade, `launchctl kickstart -k
gui/$(id -u)/ie.lazzurs.whisper-server`.

A LaunchAgent only runs while the user is logged in, hence the next step. (A
system LaunchDaemon would start without a login, but GPU access from a
daemon outside a user session is less predictable; the agent plus automatic
login is the dependable setup.)

## 5. Log in automatically

System Settings → Users & Groups → **Automatically log in as** → your user.
macOS only offers this with **FileVault off** (with FileVault on, the disk
must be unlocked by hand after every boot, which defeats the point on a
headless box). Weigh that against the Mac's physical security; with FileVault
on, expect to type the password after each power cut.

Also turn off anything that blocks an unattended session: lock-screen
password after screen saver can stay on (the LaunchAgent keeps running while
the screen is locked).

## 6. Never sleep; come back after a power cut

```sh
sudo pmset -a sleep 0          # never sleep the system
sudo pmset -a disksleep 0
sudo pmset -a displaysleep 10  # the display may sleep; that is fine
sudo pmset -a powernap 0
sudo pmset -a womp 1           # wake for network access
sudo pmset -a autorestart 1    # start up again after a power failure
pmset -g                       # check
```

`autorestart` applies to desktops (Mac mini, iMac, Mac Studio). A MacBook
restarts when power returns only if it was shut down by the battery running
out, and should stay on its charger with the lid open (or in clamshell mode
with an external display) to avoid sleeping.

In System Settings → Energy (or Battery → Options on a laptop), also enable
"Prevent automatic sleeping when the display is off" and "Start up
automatically after a power failure" where offered; they map to the same
settings.

## 7. Check from Ringmaster's side

Send a test recording with a double-click-hold. In the audit log
(`audit` table, kind `transcript`) the `source` is `primary` when the Mac
answered, `fallback` for the second Mac, and `phone` when neither did and the
phone's own transcription was used. When the Mac is asleep and the phone sent
no transcription, the job waits and retries (you get one "delayed"
notification), up to the profile's `job_max_age_secs`.
