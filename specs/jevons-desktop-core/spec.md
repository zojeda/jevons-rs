# jevons-desktop-core

## Purpose

`jevons-desktop-core` is the platform-free core of the jevons desktop app. It turns a take (what
the user said, and the window they said it in) into an action: text typed where they were, an
answer in the bubble, a tool call or an automation run. The flow tree routes each take, machines
keep tasks waiting across takes, and XPath extracts and the context investigator read the screen.
Every operating system service sits behind a trait in `platform`, so the same core runs on each
platform, in headless runs and in tests.

## Scope

It owns:

- the platform traits (`platform`) and the context snapshot with its privacy limits (`context`);
- the flow tree in `flow/`: node files, the loader and its checks, guards, the walker, machines,
  extracts, the context investigator, tool loops, tools and confirmations;
- XPath over accessibility trees (`xpath/`), the inspector's interface browser (`interface`) and
  recorded interfaces (`recorded`);
- automations (`automation/`) and recorded demonstrations (`recording/`);
- the settings folder, its git repository and history clears (`settings`, `git`, `history`);
- the take pipeline (`pipeline`), paste safety (`delivery`) and the typed API client (`client`);
- the model catalog and downloads (`catalog`, `download`), the microphone meter (`levels`) and
  the tray icon frames (`icons`).

It leaves:

- each platform's implementation of the traits (UI Automation, SendInput, CPAL, the tray and
  hotkeys), the inspector and settings windows, and the embedded runtime to `jevons-desktop`;
- transcription, decisions and generation to the jevons API (`jevons-api` and the services below
  it), which it reaches only over HTTP through `client`. It depends on no other workspace crate
  but `jevons-audio`.

## Capabilities

| Capability | Covers |
| --- | --- |
| [flows](flows.md) | Node files, the loader, guards and `[prefer]`, placeholders, shapes, the walker, the built-in tree and its upgrades |
| [machines](machines.md) | Machine folders (`root.toml`, `task.toml`) and their checks, the runtime across takes and timers |
| [extract](extract.md) | The XPath subset and its evaluation, `[extract]` reads, the workbench, selectors and the interface browser |
| [investigator](investigator.md) | The context investigator, its navigation tools, its path cache and its limits |
| [tools](tools.md) | The tool host: built-in tools, MCP servers, confirmations and tool loops |
| [automations](automations.md) | The automations library, the sandboxed engine, checks, dry runs, approval and authoring |
| [recording](recording.md) | Recording a demonstration of a task, and recorded interfaces |
| [settings](settings.md) | The settings folder, its defaults, its git repository, the reset and history clears |
| [pipeline](pipeline.md) | One take from the microphone to delivery, live dictation and paste safety |
| [client](client.md) | Typed requests to the jevons API, the API log, the model catalog and downloads, tray icon frames |
