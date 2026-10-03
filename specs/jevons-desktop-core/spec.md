# jevons-desktop-core

## Purpose

`jevons-desktop-core` is the platform-free client of the jevons desktop app: what sits at the
user's desk. It captures what is on screen, reads the interface of the user's applications, types
text where they were, asks before a tool runs, and runs automations on those applications. Every
operating system service sits behind a trait in `platform`, so the same client runs on each
platform, in headless runs and in tests. What to do with a take is the server's,
`jevons-desktop-server`.

## Scope

It owns:

- the platform traits (`platform`);
- XPath over accessibility trees (`xpath/`), reading `[extract]` expressions (`reader`), the
  inspector's interface browser (`interface`) and recorded interfaces (`recorded`);
- the desk the server calls, in-process (`desk`): delivery with paste safety (`delivery`),
  confirmations (`confirm`), and the investigations' elements and remembered paths (`look`);
- automations (`automation/`) and recorded demonstrations (`recording/`);
- the client's settings (`config`), the settings folder's git repository (`git`), the guides
  jevons keeps in a folder (`guarded`) and history clears (`history`);
- the model catalog and downloads (`catalog`, `download`), the microphone meter (`levels`) and
  the tray icon frames (`icons`).

It leaves:

- each platform's implementation of the traits (UI Automation, SendInput, CPAL, the tray and
  hotkeys), the inspector and settings windows, the settings folder as a whole and the embedded
  runtime to `jevons-desktop`;
- the flow tree, the machines, the pipeline, the inference routes and the model loops to
  `jevons-desktop-server`. It does not depend on it: both speak in the types of
  [jevons-desktop-protocol](../jevons-desktop-protocol/spec.md), and the automation author gets
  its model as a `Planner` the app implements.

## Capabilities

| Capability | Covers |
| --- | --- |
| [desk](desk.md) | What the server asks of the client, carried out with the platform layers: delivery, confirmation, reading, investigations, the client's tools |
| [extract](extract.md) | The XPath subset and its evaluation, `[extract]` reads, the workbench, selectors and the interface browser |
| [automations](automations.md) | The automations library, the sandboxed engine, checks, dry runs, approval and authoring |
| [recording](recording.md) | Recording a demonstration of a task, and recorded interfaces |
| [models](models.md) | The model catalog and downloads, tray icon frames |
