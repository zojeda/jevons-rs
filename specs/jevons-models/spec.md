# jevons-models

## Purpose

`jevons-models` tells which architecture a model's files hold and loads it. Diffusion models load as
a `DiffusionModel` (DiffusionGemma, Nemotron-Labs-Diffusion) and speech models as a `SpeechModel`
(Parakeet TDT), on the calling thread. It names the model IDs and routing aliases the server serves
by default, and adapts the DiffusionGemma runtime to the `DiffusionModel` contract.

## Scope

It owns:

- `detect`: architecture detection from a GGUF file or a Hugging Face checkpoint directory, `--arch`
  resolution, default model IDs and `-latest` aliases.
- `speech`: speech detection, default speech model IDs and `load_speech`.
- `gemma4`: the DiffusionGemma adapter. Its chat markers, answer-code pieces, image limits and
  prompt-cache switch live here; the GPU runtime under it is `jevons-gemma4-diffusion`.

It leaves to other crates:

- Model configuration and its validation to `jevons-core` (`ModelConfig`, `SpeechConfig`).
- GGUF reading to `jevons-formats`.
- The runtimes to `jevons-gemma4-diffusion`, `jevons-nemotron-diffusion` and `jevons-parakeet`. The
  Nemotron and Parakeet crates implement their contracts themselves.
- Samplers and the answer-code check to `jevons-diffusion`; served names and aliases to
  `jevons-api`.

Principles P1 and P2 apply: the crate forbids unsafe code, and loading runs on the caller's thread,
which is the model's worker thread.

## Requirements

### R1 Architecture names

The diffusion architectures are `gemma4-diffusion` and `nemotron-diffusion`. `--arch` accepts either
ID or `auto`. Any other name fails with `InvalidInput`, listing `auto` and the known IDs.

Tests: `architecture_names_round_trip_and_unknown_names_are_rejected`

### R2 GGUF files

A file that starts with the `GGUF` magic and whose `general.architecture` is `diffusion-gemma` is
DiffusionGemma. A GGUF with another architecture or none fails with `UnsupportedModel` naming it. A
file without the magic fails with `UnsupportedModel`, and a file that cannot be opened or read fails
with `ModelLoad`.

Tests: `files_without_gguf_magic_are_unsupported`

### R3 Checkpoint directories

A path that names a directory, its `config.json`, a `.safetensors` file or a
`.safetensors.index.json` file resolves to the checkpoint directory. Its `config.json` is
Nemotron-Labs-Diffusion when `model_type` is `nemotron_labs_diffusion` or
`nemotron_labs_diffusion_vlm`, or when `architectures` lists `NemotronLabsDiffusionModel` or
`NemotronLabsDiffusionVLMModel`. Other model types fail with `UnsupportedModel`. A missing or
malformed `config.json` fails with `ModelLoad`.

Tests: `checkpoint_directories_are_detected_from_config_json`

### R4 Speech checkpoints

`detect_speech` accepts a checkpoint directory whose `model_type` is `parakeet_tdt`, as architecture
`parakeet-tdt`. Other model types, and paths that are not checkpoints, fail with `UnsupportedModel`.
Asking `detect` for a Parakeet checkpoint fails with `UnsupportedModel` pointing to
`services.speech`.

Tests: `parakeet_checkpoints_are_speech_models_and_llms_are_not`

### R5 An explicit architecture must match the files

When the configuration names an architecture other than `auto`, it must equal the detected one. A
mismatch fails with `UnsupportedModel` naming both.

Tests: none yet

### R6 Default model IDs

DiffusionGemma serves as `gemmadiffusion-0.1`. A Nemotron checkpoint serves as
`nemotron-diffusion-3b`, `-8b` or `-14b` when its layer count and hidden size are (26, 3072), (34,
4096) or (40, 5120), and as `nemotron-diffusion-8b` for any other shape or an unreadable
`config.json`. A Parakeet checkpoint with vocabulary 8193 and 24 encoder layers serves as
`parakeet-tdt-0.6b-v3`, and any other as `parakeet-tdt`.

Tests: `checkpoint_directories_are_detected_from_config_json`, `parakeet_checkpoints_are_speech_models_and_llms_are_not`

### R7 Aliases and projectors

The `-latest` aliases are `gemmadiffusion-latest`, `nemotron-diffusion-latest` and
`parakeet-latest`. DiffusionGemma is the one architecture that takes a separate vision projector
file (`mmproj`).

Tests: none yet

### R8 Loading

`load` validates the `ModelConfig` (`jevons-core`), resolves the architecture and loads the model on
the calling thread. Nemotron and Parakeet receive the checkpoint directory even when the path names
a file inside it.

Tests: none yet

### R9 Feature gates

The features `gemma4`, `nemotron` and `parakeet` (all on by default) build each runtime in.
Detection covers every architecture; loading one whose feature is off fails with `UnsupportedModel`:
"support is not built into this binary".

Tests: none yet

### R10 DiffusionGemma chat format

DiffusionGemma prompts start with BOS and use `<|turn>user\n`, `<turn|>\n<|turn>model\n` and
`<|turn>system\n`, with `<turn|>\n` closing a turn. The empty thought is
`<|channel>thought\n<channel|>`. Answers stop at `<turn|>`, `<eos>` or `<pad>`, and thoughts at
`<channel|>` or `<turn|>`. The scheme is uniform noise with self-conditioning, excluding the mask
token. A canvas holds at most 64 tokens.

Tests: `model_reads_preserve_reproducibility_across_requests`

### R11 DiffusionGemma load checks

Loading fails with `UnsupportedModel` when the GGUF's tokenizer cannot be built or its vocabulary
size differs from the model's. Runtime errors map to `InvalidInput` for bad input, `ModelLoad` for
unreadable GGUF files and `Backend` for checkpoints or projectors outside the supported kernel set.
The kernels are tuned and warmed before `load` returns, so the first request pays no compilation.

Tests: none yet

### R12 Answer-code pieces

A DiffusionGemma token can be an answer code when it lies in the vocabulary, is not a control token,
and its piece, after one leading space, is 1 to 16 ASCII letters or digits. The adapter takes the
rule from `jevons-tokenizer` (R15 there).

Tests: `code_pieces_ignore_one_leading_space_and_skip_control_tokens`

### R13 DiffusionGemma images

Image input needs `mmproj`; without it a request with images fails with `InvalidInput` naming
`--mmproj`. A request holds at most 8 images, and an image whose tokens exceed `batch_size` fails
with `InvalidInput` asking for a larger `--batch-size`. Each image becomes `<|image>`, its rows and
`<image|>`; loading fails when either marker is not a single token. Image rows are keyed by the
image's size and pixels, so a repeated image reuses its prompt rows and reproduces the read bit for
bit.

Tests: `model_extensions_average_refine_think_and_chunk`, `model_images_prefill_and_preserve_text_reproducibility`

### R14 DiffusionGemma prompts and cache

A prompt longer than `context_size` fails with `InvalidInput` before any GPU work. With
`prompt_cache` on, a prefill reuses the longest resident prefix; with it off, each prefill
recomputes the whole prompt. The prefill profile counts calls, batches, processed and reused tokens,
and processed plus reused tokens equal the prompt length.

Tests: `model_reads_preserve_reproducibility_across_requests`
