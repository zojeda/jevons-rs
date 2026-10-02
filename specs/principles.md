# Principles

Rules every crate follows. A spec may add to them, never relax them.

## Code

- **P1 No unsafe code.** The inference crates forbid it, CubeCL kernels launch in checked mode, and
  desktop platform code goes only through safe wrapper crates.
- **P2 Models stay on their worker thread.** Blocking inference never runs on a Tokio executor
  thread.
- **P3 Wire formats live in `jevons-api`.** Services (`jevons-generative`, `jevons-decision`,
  `jevons-speech`) take no HTTP or async code and parse no request bodies. JSON appears in them
  only as data: tool arguments, schemas and structured answers as `serde_json::Value` in
  `jevons-generative`, serde derives on `jevons-decision`'s read types, and JSON output in the
  `jevons-scm` CLI and the examples.
- **P4 No backward compatibility.** When a design replaces another, the old one goes: no shims,
  no accepted-and-ignored fields. An unedited built-in flows folder is brought up to date by
  `flow/earlier.rs`.
- **P5 Each layer depends only on the layers below it,** as the README's Architecture diagrams
  show.

## Privacy

- **P6 Nothing logs what you said or what your screen showed.** Snapshots, transcripts and keys
  stay out of logs and traces' log lines. The API log (`privacy.log_api`, off by default) is the
  only exception, and it never writes keys.
- **P7 Context reads stay where you allow them.** Password fields are never read, text is capped
  at `privacy.max_context_chars`, and other windows are read only when the settings name them.

## Desktop safety

- **P8 A tool asks before it runs** unless the settings say otherwise, and tools come only from the
  settings, never from the flows folder.
- **P9 Text goes only into the window the take started in,** once no key is held; otherwise it
  waits on the clipboard.
- **P10 An unsure take never moves a task on.**
- **P11 A machine calls only the tools its `tools` list names.**
- **P12 An automation runs only in the exact version you approved.**

## Performance

- **P13 Dictation costs one decision call per take,** through every routing level.
