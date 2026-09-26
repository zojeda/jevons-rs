# jevons-desktop

Context-aware dictation from the tray. Press the hotkey in any application and speak. jevons-desktop reads where you are, transcribes you, picks a profile for that place, decides what to do with your words, and types the result into the field you started in.

It runs the jevons models in-process, optionally exposing the API on a port, or uses a jevons server elsewhere. The platform-free behaviour lives in [`jevons-desktop-core`](../jevons-desktop-core). This crate adds the tray, the window and the per-OS layers. The full guide is [docs/desktop.md](../../docs/desktop.md).

## From speech to inserted text

```mermaid
flowchart TD
    press(["Hotkey pressed<br/>tap to toggle · hold to dictate"])
    press --> snapshot["Context snapshot<br/>app · window · URL<br/>field role and name<br/>selection · text around the caret"]
    press --> mic["Microphone<br/>CPAL → 24 kHz PCM16<br/>100 ms chunks"]

    snapshot --> privacy["Privacy limits<br/>no password text · truncation<br/>clipboard only if allowed"]

    mic --> realtime{"/v1/realtime<br/>served?"}
    realtime -- yes --> stream["Stream audio<br/>live text in the inspector"]
    realtime -- "no or failed" --> buffer["Buffer the take"]
    release(["Hotkey released<br/>or tapped again"]) --> commit
    stream --> commit["Commit the turn"]
    buffer --> upload["/v1/audio/transcriptions"]
    commit --> transcript["Transcript"]
    upload --> transcript

    privacy --> resolve["Resolve the profile<br/>rules: app · title · URL · role · field name<br/>highest priority, then most specific<br/>then the best destination"]
    transcript --> decide
    resolve --> decide{"/v1/systemone<br/>asks only what is open"}

    decide -- "action?<br/>when the profile says auto<br/>and the field has text" --> action["insert · replace · rewrite"]
    decide -- "needs_generation?" --> needs{"above the<br/>threshold?"}
    decide -- "profile?<br/>when two profiles tie" --> resolve

    action --> needs
    needs -- "no: clean dictation" --> raw["Use the transcript as heard"]
    needs -- "yes, or rewrite" --> generate["/v1/responses, streamed<br/>base + action + profile<br/>+ destination instructions"]

    raw --> safe
    generate --> safe{"Same window?<br/>All keys released<br/>within 2 s?"}
    safe -- yes --> deliver["Deliver<br/>paste (clipboard restored) · type<br/>· set value · copy"]
    safe -- no --> clipboard["Leave it on the clipboard<br/>and say why"]

    deliver --> trace[("Trace<br/>context · rules checked · probabilities<br/>prompt · output · timings")]
    clipboard --> trace
```

1. **At the press** the app captures the context, before your focus can move, and opens the microphone. The context comes from UI Automation on Windows; other platforms read only the active window for now.
2. **While you speak** the audio streams to Realtime transcription, and the inspector shows live text. When Realtime is not served, the take is uploaded when you finish.
3. **The profile** comes from rules you write in TOML. The inspector shows every rule it checked against the real context, so a mismatch is easy to spot.
4. **The decision model** answers only the open questions: which action (only when the profile says `auto` and there is text to act on), whether the words need editing, and which profile when two tie. Clean dictation skips generation entirely.
5. **Generation** follows the base prompt, then the action, then the profile's and the destination's instructions.
6. **Delivery** goes only into the window the take started in, once every key is released. Otherwise the text waits on the clipboard.

## Threads

```mermaid
flowchart LR
    tray["Tray thread · tao<br/>icon animation · menu · global hotkey"] -- commands --> agent
    window["Main thread · eframe<br/>inspector · settings · models"] <-- "view · commands" --> agent
    agent["Agent thread<br/>gestures · context · takes"] --> takes["Takes on Tokio workers<br/>pipeline"]
    takes -- "HTTP · WebSocket" --> runtime["Runtime thread<br/>jevons_api::load + serve<br/>or a remote server"]
    takes --> sink["TextSink<br/>SendInput · clipboard"]
```

## Run

```bash
cargo run --release --locked -p jevons-desktop                 # tray, window, embedded models
cargo run --release --locked -p jevons-desktop --no-default-features   # remote server only
cargo run -p jevons-desktop -- --replay examples/speech-en.flac \
  --context examples/desktop/context-slack.json               # one take, trace as JSON
```

On Linux the build needs `libgtk-3-dev libxdo-dev libayatana-appindicator3-dev libasound2-dev libssl-dev`.

To build the Windows app from WSL without the embedded runtime, use [cargo-xwin](https://github.com/rust-cross/cargo-xwin), with `clang-cl` and `lld-link` on `PATH`:

```bash
PATH=/usr/lib/llvm-18/bin:$PATH cargo xwin build --release -p jevons-desktop \
  --no-default-features --target x86_64-pc-windows-msvc
```

## Layout

| Module | Role |
| --- | --- |
| `agent` | Hotkey gestures, context capture, takes, trace history, the shared view |
| `tray` | tao event loop: tray icon frames, menu, global hotkey |
| `audio` | CPAL capture: downmix, resample to 24 kHz, 100 ms chunks, meter |
| `runtime` | Embedded models (private loopback or exposed port) or a remote server |
| `platform` | Per-OS `ContextProvider` and `TextSink`; `windows` uses UI Automation, enigo and the clipboard |
| `ui` | Context and resolution, takes, profiles, settings, models (catalog and downloads) |
