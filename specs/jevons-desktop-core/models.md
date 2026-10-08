# jevons-desktop-core: models

[Back to jevons-desktop-core](spec.md)

The model catalog lists what the app can download from Hugging Face, and downloads fetch, resume
and verify it. The tray icon's frames sit here too. Which model serves which capability is the
server's ([client](../jevons-desktop-server/client.md)) and the app's
([runtime](../jevons-desktop/runtime.md)).

## Requirements

### R1 The catalog lists the models the app can download

The built-in catalog holds, in order: DiffusionGemma 26B-A4B Q4_K_M (generative and decision, from
`unsloth/diffusiongemma-26B-A4B-it-GGUF`, with its vision projector from
`FreedomAISVR/DiffusionGemma-26B-A4B-it-MXFP4-GGUF`), Nemotron-Labs-Diffusion 3B and VLM 8B
(generative and decision), and Parakeet TDT 0.6B v3 (speech). Each entry downloads into a folder
named by its id under the models folder, and loads as its model file or as that folder.

Tests: `user_entries_extend_and_override_the_builtin_catalog`, `downloaded_catalog_models_fill_unset_services_with_gemma_first`

### R2 `models.toml` extends the catalog

Entries in `models.toml` next to the settings file add to the built-in catalog, and an entry with
a built-in id replaces that one. Unknown fields are errors. A file that does not parse leaves the
built-in catalog, with the error.

Tests: `user_entries_extend_and_override_the_builtin_catalog`

### R3 Downloaded models fill the services nobody chose

A service with no model selected uses the first downloaded catalog entry that serves it, in
catalog order: DiffusionGemma first for generative and decision, then Parakeet for speech. A
selected model stays. With `runtime_config` set, nothing is filled.

Tests: `downloaded_catalog_models_fill_unset_services_with_gemma_first`

### R4 Downloads fetch only what the entry names

A download lists the repository's files at the entry's revision (and those of its extra sources)
and keeps the ones its globs match; none matching is an error. `HF_TOKEN`, when set, goes with
every request. Nothing downloads unless the user asks.

Tests: `interrupted_download_resumes_with_range`, `extra_sources_download_into_the_same_folder`

### R5 Downloads resume and verify

Each file downloads into `<file>.part`, resuming an interrupted one with an HTTP range. A file with
a SHA-256 in the repository is checked against it: a mismatch removes the part file, keeps no
final file and fails the download. A finished file is renamed into place, and once every file is
in place a marker makes the entry ready.

Tests: `interrupted_download_resumes_with_range`, `checksum_mismatch_marks_model_corrupt_and_keeps_no_final_file`

### R6 Downloads can be cancelled and never run twice

A cancelled download stops, keeps its part file for a later resume, and leaves the entry not
ready. A model folder that is being downloaded cannot start a second download in the same
process.

Tests: `a_cancelled_download_keeps_its_part_file`, `a_model_being_downloaded_cannot_be_downloaded_twice`

### R7 Downloads stay in their folder

A repository path that is absolute or climbs out with `..` is refused, so no file lands outside
the model's folder.

Tests: `repository_paths_cannot_escape_the_model_folder`

### R8 Every tray state has its own frame

The tray icon is the jevons alien at 32 pixels: cyan when ready, blue while the models load, grey
with no model loaded, amber while GPU kernels are tuned, red after a failed take, and magenta
while a demonstration is recorded. Listening shows a waveform with one frame per meter level,
louder levels drawing taller bars. Transcribing (green) and deciding and writing (violet) show
dots that cycle through three frames, 180 ms apart. Each state indexes its own frame and has its
own tooltip.

Tests: `every_state_indexes_its_own_frame`, `tuning_has_its_own_amber_frame_and_explains_itself`, `louder_audio_draws_taller_bars`, `processing_states_cycle_through_three_frames`

### R9 The app icon glows at the eyes

The app icon, drawn at any size, is a dark head with glowing eyes and clear corners.

Tests: `the_app_icon_glows_at_the_eyes_on_a_dark_head_with_clear_corners`
