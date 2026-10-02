# Settings

[jevons-api](spec.md)

## Purpose

The settings file declares the models to load and the services that use them, and command-line
flags override parts of it. `jevons.example.toml` shows every key.

## Scope

Loading the declared models belongs to [workers](workers.md). The binary that parses the
process arguments is `jevons-rs`.

## Requirements

### R1 Where the file comes from

The file is `--config PATH`, or `JEVONS_CONFIG` when the flag is absent. Without either, it is
`./jevons.toml` if that file exists, then `$XDG_CONFIG_HOME/jevons/config.toml` (or
`~/.config/jevons/config.toml`). With none of these, loading fails with a message that names
the three options. A file that cannot be read fails with its path in the message.

Tests: `flags_override_the_file_and_a_missing_file_is_an_error`

### R2 Sections and keys

The file has three sections, and every key outside this table is an error:

| Section | Keys and defaults |
| --- | --- |
| `[server]` | `bind` (`127.0.0.1:8080`), `api_key` (none) |
| `[models.<name>]` | `path` (required), `id` (the model's own), `main_gpu` (0), `queue_capacity` (8), `arch` (detected), `mmproj` (none), `context_size` (8192), `batch_size` (512), `prompt_cache` (true), `seed` (42), `decoding` (`diffusion`) |
| `[services.generative]` | `model` |
| `[services.decision]` | `model` |
| `[services.speech]` | `model`, `max_audio_seconds` (3600.0), `realtime` (true) |

`decoding` takes `diffusion`, `self-speculation` or `autoregressive`. `arch` takes `auto`,
`gemma4-diffusion` or `nemotron-diffusion`; at startup, a named architecture that differs from
the one detected in the files is an error.

Tests: `services_share_a_model_declared_once`, `generative_and_decision_may_use_different_models`, `inconsistent_settings_are_errors`

### R3 Paths

`path` and `mmproj` resolve against the settings file's directory when relative. A leading `~/`
resolves against `HOME`, or `USERPROFILE` when `HOME` is unset. Absolute paths stay as written.

Tests: `services_share_a_model_declared_once`, `home_relative_paths_expand`

### R4 Services and models must agree

Loading fails with `Invalid settings file <path>: ...` when:

- no service is enabled;
- a service names a model with no `[models.<name>]`;
- a declared model serves no service;
- one model serves both Speech and Generative or Decision;
- `max_audio_seconds` is not a positive finite number;
- a model's `queue_capacity` is 0 or its `id` is blank;
- a model's `decoding` is not one of the three names;
- `bind` is not a socket address, or the TOML does not parse.

Tests: `inconsistent_settings_are_errors`

### R5 Services share a declared model

Generative and Decision may name the same model or two different ones. Each distinct diffusion
model is loaded once, whatever the number of services that name it.

Tests: `services_share_a_model_declared_once`, `generative_and_decision_may_use_different_models`

### R6 Flags override the file

`--bind ADDR` replaces `server.bind`. `--api-key KEY`, or `TYPESAFE_API_KEY` when the flag is
absent, replaces `server.api_key`. An empty key from any source fails loading. Unknown flags are
errors.

Tests: `flags_override_the_file_and_a_missing_file_is_an_error`

### R7 Diffusion keys stay off speech models

`arch`, `mmproj`, `context_size`, `batch_size`, `prompt_cache`, `seed` and `decoding` apply to
diffusion language models. A speech model that sets any of them fails at startup with a message
that names the keys it set.

Tests: none yet
