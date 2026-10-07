# Workers

[jevons-api](spec.md)

## Purpose

Loading the configured models, the names they answer to, the queues in front of them, and the
server's lifecycle around them.

## Scope

Architecture detection and each architecture's default ID come from `jevons-models`; the
engines themselves from `jevons-diffusion` and `jevons-speech`.

## Requirements

### R1 One worker per loaded model

Startup loads each model that a service names on its own thread, which owns the model until the
thread ends. When Generative and Decision name the same model, one engine, one thread and one
queue serve both. Two different models get a thread and a queue each. Loading finishes before
the server answers its first request.

Tests: `services_share_a_model_declared_once`, `generative_and_decision_may_use_different_models`

### R2 Served IDs

A model's served ID is its `id` from the settings. Without one, it is the model's default:
`gemmadiffusion-0.1` for DiffusionGemma, `nemotron-diffusion-3b`, `-8b` or `-14b` for
Nemotron-Labs-Diffusion by checkpoint size (`nemotron-diffusion-8b` when the size is unknown),
and `parakeet-tdt-0.6b-v3` for Parakeet TDT v3 (`parakeet-tdt` for other Parakeet TDT
checkpoints).

Tests: none yet

### R3 Aliases

Besides its served ID, each model answers to its architecture's alias: `gemmadiffusion-latest`,
`nemotron-diffusion-latest` or `parakeet-latest`. The diffusion model that serves Decision, or
the only diffusion model when Decision is off, adds `openjev-latest` and `jev-latest`. An
alias equal to the served ID is dropped.

Tests: none yet

### R4 Names are unique

Startup fails with `Two models answer to "<name>"; give them different ids` when two loaded
models share a served ID or an alias.

Tests: none yet

### R5 Load failures stop startup

A model whose architecture cannot be detected stops startup with an error that starts with
`models.<name>:`. A model that fails to load stops startup with the engine's error, and so does
a `decoding` other than `diffusion` on a model without masked causal predictions, such as
DiffusionGemma. An `mmproj` on an architecture whose vision tower is in the model files is
dropped with a warning.

Tests: none yet

### R6 Bounded queues

Each diffusion model's queue holds `queue_capacity` waiting jobs. A request that finds the queue
full gets `529` (`overloaded_error`, `The inference queue is full. Retry later.`) with
`retry-after: 1`. A request that finds the worker gone gets `503` (`overloaded_error`,
`The inference worker is unavailable`).

Tests: `queue_saturation_and_worker_failure_are_reported`

### R7 A lost worker fails its waiting requests

When a worker stops after it accepted a job, the waiting request gets `503`, and the worker
counts as dead for `/health`.

Tests: `worker_lost_after_accepting_a_job_returns_unavailable`

### R8 One job at a time

A diffusion worker runs one job at a time, in arrival order, whichever service it comes from:
a System One read, a free-form generation or a tool turn.

Tests: none yet

### R9 Gone clients cost nothing

A queued job whose client has disconnected is skipped. A generation stops at the next decided
text once its client disconnects. A transcription stops after the current window.

Tests: none yet

### R10 Speech queues put live audio first

The speech worker has two queues: uploads, holding `queue_capacity` jobs, and live Realtime
jobs, holding 16 across every session. A full upload queue answers as in R6. Live jobs run
before waiting uploads and between the windows of a long upload, so a Realtime session never
waits for a whole recording.

Tests: `live_passes_run_between_the_windows_of_a_long_recording`

### R11 Speech jobs

An upload job streams each closed segment with its words, then the transcript. A live pass
transcribes at most one model window and returns its words and text.

Tests: `recordings_stream_segments_then_the_transcript`, `live_passes_return_words_and_text`

### R12 Serving and shutdown

`run` binds the listen address before loading any model, so a taken address fails at once. It
serves until SIGINT or SIGTERM (Ctrl-C on systems without Unix signals), finishes pending
requests, then waits for the worker threads to end and release their models. Every worker is
waited for, also after one panicked. When a model fails to load, the models loaded before it are
dropped, and loading returns its error only once their workers have ended: a caller that then
exits, or loads again, never does so while a model is still being freed on the device.

Tests: `joining_waits_for_every_worker_even_after_one_panicked`

### R13 Serving again keeps models loaded

`serve` stops when its shutdown future completes and leaves the models loaded: the same state
served on another listener answers at once. A host may give each listener its own API key.

Tests: `rebinding_the_listener_keeps_models_loaded`, `a_listener_can_override_the_api_key`
