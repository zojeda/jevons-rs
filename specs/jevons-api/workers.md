# Workers

[jevons-api](spec.md)

## Purpose

Loading the configured models, the names they answer to, the queues in front of them, and the
server's lifecycle around them.

## Scope

Architecture detection and each architecture's default ID come from `jevons-models`; the
engines themselves from `jevons-diffusion`, `jevons-speech` and `jevons-voice`.

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
`parakeet-tdt-0.6b-v3` for Parakeet TDT v3 (`parakeet-tdt` for other Parakeet TDT checkpoints),
and `qwen3-tts-12hz-0.6b-base` or `qwen3-tts-12hz-1.7b-base` for Qwen3-TTS Base by checkpoint
size (`qwen3-tts` when the size is unknown).

Tests: none yet

### R3 Aliases

Besides its served ID, each model answers to its architecture's alias: `gemmadiffusion-latest`,
`nemotron-diffusion-latest`, `parakeet-latest` or `qwen3-tts-latest`. The diffusion model that
serves Decision, or
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
DiffusionGemma, and a `services.voice.voice` the voice model does not list. An `mmproj` on an
architecture whose vision tower is in the model files is dropped with a warning. R17 says when
a failure leaves the other models up.

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
waited for, also after one panicked. When a model fails to load and stops the load, the models
loaded before it are dropped, and loading returns its error only once their workers have ended:
a caller that then exits, or loads again, never does so while a model is still being freed on
the device.

Tests: `joining_waits_for_every_worker_even_after_one_panicked`

### R13 Serving again keeps models loaded

`serve` stops when its shutdown future completes and leaves the models loaded: the same state
served on another listener answers at once. A host may give each listener its own API key.

Tests: `rebinding_the_listener_keeps_models_loaded`, `a_listener_can_override_the_api_key`

### R14 The voice queue

The voice worker has one queue of `queue_capacity` waiting speeches. A full queue answers as in
R6, and so does a worker that is gone. A worker that stops after it accepted a speech, before
any audio, gives the waiting request `503`. The worker speaks one job at a time, in arrival
order.

Tests: `a_full_voice_queue_is_overloaded_and_a_stopped_worker_unavailable`, `a_full_voice_queue_answers_529_and_a_stopped_worker_503`, `a_voice_worker_lost_before_its_first_audio_answers_503`, `the_voice_worker_speaks_one_job_at_a_time_in_arrival_order`

### R15 Voice jobs

A voice job streams the model's audio in stretches, mono at the model's rate, as the model
produces them, then what was spoken: the sentences, the samples, and whether a listener stopped
it. The request's voice and language reach the Voice service, which speaks in its default voice
and guesses each sentence's language without them. A text the service refuses (a voice or a
language the model lacks, nothing to say) ends the job with that error before any audio. A
model failure ends the job after the audio of the sentences before it.

Tests: `a_speech_streams_audio_chunks_then_what_was_spoken`, `a_text_the_model_cannot_speak_fails_before_any_audio`

### R16 A listener that leaves stops the voice model

A queued speech whose client has gone is skipped. A running one stops at the model's next
stretch of audio: the model is asked for no further sentence, and the next job starts.

Tests: `a_listener_that_leaves_stops_the_model_at_its_next_chunk`, `a_client_that_hangs_up_stops_the_model`

### R17 Optional services

`load_optional(settings, optional)` loads as `load` does, except that the services in `optional`
may fail. When a model fails to load and every service that names it is in `optional`, the load
goes on without it: the state serves the other services, `/health` reports the failed ones as
off, and `Loaded.unserved` returns each with its error for the caller to show. A failed model
that also serves a service outside `optional` stops the load, as in R12. The voice model loads
last. `load`, which `run` calls, names no optional service, so the server binary refuses to
start on any failure.

Tests: `a_failed_model_is_left_out_only_when_every_service_that_names_it_is_optional`, `a_voice_model_that_fails_to_load_stops_the_load_unless_voice_is_optional`

### R18 Loading the voice model

The voice worker loads its model with the settings' default voice, or the first voice the model
lists, and speaks one short sentence before it serves, so the first request does not wait for
kernels to compile. When the model fails to load, starting the worker fails with the model's
error, once the worker's thread has ended.

Tests: `a_voice_model_that_fails_to_load_fails_the_start_once_its_thread_ended`
