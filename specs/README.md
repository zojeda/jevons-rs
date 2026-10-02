# Specs

Specs say what each part of jevons-rs does and must keep doing. `docs/` explains how to use it.
Specs are the source of truth for behavior: change a spec when you change what the code does.

## Layout

```
specs/
  README.md                this file: the rules, the format, the index
  principles.md            rules every module follows
  <crate>/spec.md          the living spec of one crate: what it does now
  <crate>/<capability>.md  a large crate's capabilities, one file each, listed in its spec.md
  changes/<name>/          work in flight
    proposal.md            why, what changes, which specs it touches
    design.md              how (when the proposal is not enough)
    tasks.md               the checklist
```

Every crate under `crates/` has a folder here with the crate's name.

## Living specs

A living spec describes the crate as it is on the branch it lives on. It holds:

- **Purpose:** what the crate is for, in a paragraph.
- **Scope:** what it owns, and what it leaves to which other crate.
- **Requirements:** numbered, each a behavior someone can check, and the tests that check it.

A requirement looks like this:

```markdown
### R3 An unsure take stays

When the decision model's probability for a `said` transition is below `min_probability` and the
state has no `[else]`, the machine stays in its state and runs no work.

Tests: `an_unsure_take_leaves_a_waiting_task_where_it_was`
```

- Number requirements in order (`R1`, `R2` …) and never reuse a number. Mark a dropped requirement
  `Removed` instead of deleting it.
- `Tests:` names test functions, in backticks, comma-separated. A requirement no test checks says
  `Tests: none yet`.
- GPU and model tests (`#[ignore]`) count: name them like the others.
- Keep a requirement to what you can observe: inputs, outputs, errors, limits. Leave
  implementation choices to the code and to `design.md`.

## Changes

A change folder holds work that alters behavior before it lands:

- `proposal.md` says why, what changes, and which living specs it touches.
- `design.md` says how, when that needs more than the proposal.
- `tasks.md` is the checklist, ticked as work lands.

When the change lands, its behavior goes into the living specs in the same commit, and the folder
moves to `changes/archive/`.

## Keeping specs current

- **A change in behavior updates its spec in the same commit:** a new requirement, an edited one,
  or one marked `Removed`. Renaming a test updates every spec that cites it.
- **New work starts in `changes/`** when it spans more than one commit or more than one crate.
- **`scripts/check-specs.py`** fails when a spec cites a test that does not exist, when a crate has
  no spec, or when a spec is missing from the index below. CI runs it (`specs.yml`), and so does
  `just check-specs`.

## Index

| Spec | Covers |
| --- | --- |
| [principles](principles.md) | Rules every module follows |
| **Desktop** | |
| [jevons-desktop](jevons-desktop/spec.md) | The tray app: agent, runtime, inspector and bubble, platform layers, CLI |
| [jevons-desktop-core](jevons-desktop-core/spec.md) | Flows, machines, extracts, investigator, tools, automations, recording, settings, pipeline, client |
| **API** | |
| [jevons-api](jevons-api/spec.md) | HTTP routes and auth, settings, workers, OpenAI routes, tools, transcriptions, Realtime, System One |
| [jevons-rs](jevons-rs/spec.md) | The runtime binary |
| **Services** | |
| [jevons-generative](jevons-generative/spec.md) | Free-form answers, streaming, tool calls and structured answers |
| [jevons-decision](jevons-decision/spec.md) | Restricted-canvas reads and probabilities, `jevons-scm` |
| [jevons-speech](jevons-speech/spec.md) | Windowed transcription, words and segments |
| **Diffusion layer** | |
| [jevons-diffusion](jevons-diffusion/spec.md) | `DiffusionEngine`, decoding modes, samplers, the fake model |
| **Models** | |
| [jevons-models](jevons-models/spec.md) | Model detection and loading, the DiffusionGemma adapter |
| [jevons-gemma4-diffusion](jevons-gemma4-diffusion/spec.md) | DiffusionGemma on tuned CubeCL kernels |
| [jevons-nemotron-diffusion](jevons-nemotron-diffusion/spec.md) | Nemotron-Labs-Diffusion on Burn |
| [jevons-parakeet](jevons-parakeet/spec.md) | Parakeet TDT on Burn |
| **GPU runtimes** | |
| [jevons-kernels](jevons-kernels/spec.md) | Shared tuned CubeCL kernels: buffers, quantized and FP16 GEMM |
| [jevons-burn](jevons-burn/spec.md) | The Burn runtime: HIP device, weight streaming, attention, KV cache |
| **Foundation** | |
| [jevons-core](jevons-core/spec.md) | Model contracts, errors, configuration, image decoding |
| [jevons-formats](jevons-formats/spec.md) | GGUF and safetensors readers |
| [jevons-tokenizer](jevons-tokenizer/spec.md) | Gemma 4 and Hugging Face tokenizers |
| [jevons-audio](jevons-audio/spec.md) | Upload decoding, resampling, PCM16 and G.711, log-mel, voice activity |
| **Changes** | |
| [changes/desktop-server](changes/desktop-server/proposal.md) | Machines, agents and tasks; the desktop server, client and protocol |
