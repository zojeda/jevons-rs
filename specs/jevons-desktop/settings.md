# jevons-desktop: settings

[Back to jevons-desktop](spec.md)

The settings folder holds two files, `jevons-desktop.toml` (the client's settings) and
`jevons-server.toml` (the server's), and next to them unless the settings move them, the flow tree
in `flows/` and the automations library in `automations/`. The app reads both files into one
value, which the Settings tab edits, and writes each half back to its own file. jevons fills in whatever the
folder lacks, keeps it in a git repository of its own, commits what it writes there, and can put
the defaults back with the earlier settings kept in the history. The history clears remove the
logs, traces, recorded interfaces and recordings jevons keeps under `~/jevons`. The tools and MCP
servers the server's file registers are in [tools](../jevons-desktop-server/tools.md), and the
model catalog in [models](../jevons-desktop-core/models.md). The settings' types are the
client core's (`ClientConfig`) and the server's (`ServerConfig`); the folder as a whole is this
crate's.

## Requirements

### R1 Where the settings live

The client's settings file is `--config`, or `jevons-desktop.toml` in the platform configuration
folder (`%APPDATA%\jevons\config` on Windows, `~/.config/jevons` on Linux). The server's is
`jevons-server.toml` next to it. The flow tree is `flows_dir`, or `flows/` next to the file. The automations library is `[automation] dir`, or `automations/`
next to the file. `models.toml` sits next to the file. What jevons keeps of past use goes under
`~/jevons`: `logs/`, `traces/`, `trees/`, `recordings/` (or `[automation] recordings_dir`), and
`models/` (or `[models] folder`).

Tests: `approvals_are_written_alone_and_saving_the_settings_keeps_them`

### R2 A missing file means defaults, and unknown fields are errors

A settings file that does not exist loads as that half's defaults. A field its format does not
have, in any section, fails the load with that file's path and the parser's message: a section
of the server's in the client's file is one.

Tests: `a_missing_file_gives_defaults_and_unknown_fields_are_errors`, `the_example_settings_file_parses`, `the_two_halves_make_the_settings_and_back`

### R3 The files' sections and defaults

`jevons-server.toml`:

| Section | Fields and defaults |
| --- | --- |
| `[server]` | `expose` (false), `bind` (`127.0.0.1`), `port` (8080), `api_key` (none) |
| `[providers.<name>]` | `kind` (`embedded`, `jevons`, `openrouter` or `openai-compatible`), `url` (the kind's), `key` (the kind's environment variable), `extensions`, `max_questions` and `min_probability` (the kind's profile) |
| `[routes]` | `speech`, `realtime`, `decision` and `generation`, each a `provider` and an optional `model` (all `embedded`) |
| `[models]` | `folder`, `runtime_config`, `generative`, `decision` and `speech` (each a `path`, an optional `mmproj` and the `catalog` entry it came from), `realtime` (true) |
| `[privacy]` | `log_api` (false) |
| top level | `flows_dir`, `[tools.<name>]`, `[mcp.<name>]` |

`jevons-desktop.toml`:

| Section | Fields and defaults |
| --- | --- |
| `[dictation]` | `hotkey` (`Ctrl+Alt+Space`), `hotkey_mode` (`hold`), `live_hotkey` (`F9`), `live_hotkey_mode` (`hold`), `live_feedback` (true), `inspector_hotkey` (none), `branch_hotkeys` (none), `microphone` (the default device), `language` (detected), `decide` (true), `max_output_tokens` (1024) |
| `[privacy]` | `max_context_chars` (2000), `read_clipboard` (false), `read_other_windows` (false), `readable_apps` (none) |
| `[automation]` | `dir`, `recordings_dir`, `record_hotkey` (none), `hotkeys` (none), `approved` (none), `unconfirmed` (none), `allow` (none), `author_model` (the generative model) |

The example files, `jevons-desktop.example.toml` and `jevons-server.example.toml`, parse.

Tests: `the_example_settings_file_parses`, `saved_settings_load_back_unchanged`, `the_two_halves_make_the_settings_and_back`

### R4 Hotkey modes are `hold` or `toggle`

`hotkey_mode` (push-to-talk and the branch hotkeys) and `live_hotkey_mode` each take `hold` or
`toggle`. Any other value fails the load.

Tests: `hotkey_modes_are_hold_or_toggle`

### R5 A live hotkey turned off stays off

A settings file without `live_hotkey` gets `F9`. Turning it off saves `live_hotkey = ""`, which
loads back as off.

Tests: `a_live_hotkey_turned_off_stays_off_once_saved`

### R6 Saved settings load back unchanged

Saving writes the whole file, creating its folder, and loading it gives back the same settings.

Tests: `saved_settings_load_back_unchanged`

### R7 Approvals are written alone and survive other saves

Approving an automation writes its version into `[automation.approved]` and changes nothing else
in the file. Saving the settings from anywhere else keeps the approvals the file holds, so a
Settings panel opened before an approval cannot drop it.

Tests: `approvals_are_written_alone_and_saving_the_settings_keeps_them`

### R8 jevons fills in what the folder lacks

On every start, jevons writes each settings file there is none of, the built-in flow tree and its
guides ([flows](../jevons-desktop-server/flows.md)), and the automations library's guides ([automations](../jevons-desktop-core/automations.md)).
A deleted folder gets the defaults again on the next start.

Tests: `a_new_settings_folder_gets_the_defaults_in_a_repository_of_its_own`

### R9 The folder is a repository of its own

A settings folder in no git repository becomes one: branch `main`, marked `jevons.settings = true`
in its own `.git/config`, with line endings kept as written, and everything in it as the first
commit. The commit is "The defaults of jevons <version>" for a new folder, or "The settings as
they were when jevons <version> began versioning them" when a settings file was there. A start
that writes nothing commits nothing.

Tests: `a_new_settings_folder_gets_the_defaults_in_a_repository_of_its_own`

### R10 jevons commits what it writes there

When a start updates a file jevons wrote earlier, such as a guide from an earlier version, it
commits it as "jevons <version> wrote <files>". A commit message names its first three paths and
how many more. Every commit is authored by `jevons <jevons@localhost>`, unsigned, without hooks
or prompts, and on Windows without a console window.

Tests: `a_new_settings_folder_gets_the_defaults_in_a_repository_of_its_own`, `jevons_commits_only_the_paths_it_wrote`

### R11 A commit holds only the paths jevons wrote

jevons commits the files it wrote and nothing else, so the user's own edits in the folder stay
uncommitted for them to commit. Both sides of a rename count, a removed file or folder is
committed as removed, paths outside the repository are left out, and nothing changed means no
commit.

Tests: `jevons_commits_only_the_paths_it_wrote`, `status_paths_include_both_sides_of_a_rename`

### R12 jevons commits only to a repository it created

A repository jevons did not create (no `jevons.settings` mark) gets no commits. A settings folder
inside another repository's work tree, such as a dotfiles repository, is not made a repository of
its own. Without `git` on the `PATH`, the folder is not versioned. In each case a note says why.

Tests: `a_repository_jevons_did_not_create_is_never_committed_to`, `a_folder_in_another_repository_is_not_made_one_of_its_own`

### R13 A reset puts the defaults back and keeps the earlier settings

A reset commits everything in the folder as "The settings before the reset", removes every entry
but `.git`, writes the default settings, the built-in flow tree and an empty automations library
with its guides, and commits them as "Reset the settings to the defaults of jevons <version>".
The work tree is then clean, and the report names the entries removed and the commit that holds
the earlier settings.

Tests: `a_reset_puts_the_defaults_back_and_keeps_the_earlier_settings_in_the_history`

### R14 A reset touches only a folder that is jevons' own

A reset goes ahead in the platform's settings folder, or in a folder that holds nothing but the
two settings files, `flows/`, `automations/` and `.git`. Any other folder is left as it is, and the
error names the entries that are not jevons'.

Tests: `a_folder_holding_more_than_jevons_settings_is_not_reset`

### R15 A reset keeps what it cannot save or does not own

When the commit of the earlier settings fails, nothing is reset. A flow tree or library that the
settings moved outside the folder is left as it is, with a note, and the settings then use the
folders inside it. A folder that was not a repository becomes one once reset, as on a start.

Tests: none yet

### R16 History clears remove only jevons' files

Clearing a kind of history removes, from that kind's folder, only what jevons writes there:
`.log` files in `logs/`, `.json` files in `traces/` and `trees/`, and folders holding
`recording.json` in the recordings folder. Anything else stays and is reported as not jevons'.
The models folder is never a kind of history.

Tests: `clearing_recordings_removes_only_recording_folders`, `clearing_the_logs_empties_the_open_one_and_leaves_other_files`

### R17 The open log is emptied, not removed

Clearing the logs empties `jevons-desktop.log` instead of removing it, so a running app goes on
appending to it. An empty log counts as nothing to clear.

Tests: `clearing_the_logs_empties_the_open_one_and_leaves_other_files`

### R18 A clear reports what it did

Each clear reports its kind and what it did: `<kind>: <n> removed`, then `, <m> emptied`, then
`(kept <names>, not jevons')`, then `; could not remove <name>: <error>`, or `<kind>: nothing to
clear` when there was nothing of jevons' there.

Tests: `clearing_the_logs_empties_the_open_one_and_leaves_other_files`, `clearing_recordings_removes_only_recording_folders`

### R19 Providers say where inference may come from

`[providers.<name>]` names a provider and its `kind`. `embedded`, the models the app loads, is
always there without being written.

| Kind | Address when `url` is left out | Key when `key` is left out |
| --- | --- | --- |
| `embedded` | the app's own | the app's own |
| `jevons` | `http://127.0.0.1:8080` | `TYPESAFE_API_KEY` |
| `openrouter` | `https://openrouter.ai/api` | `OPENROUTER_API_KEY` |
| `openai-compatible` | none: `url` is required | none |

`url` is the server root, without `/v1`; a trailing `/` is dropped. In `key`, `${env:NAME}` is that
environment variable, so the key stays out of the file; an empty key counts as none.

Tests: `each_capability_goes_to_the_provider_its_route_names`, `a_provider_s_key_comes_from_the_file_or_the_environment`

### R20 Each capability goes to the provider its route names

`[routes]` sends `speech`, `realtime`, `decision` and `generation` each to a provider, with the
model to ask it for. A route left out goes to `embedded`. `realtime` left out follows `speech` when
that is on an embedded or jevons provider, and is off otherwise. An embedded or jevons provider
names its own model when the route has none. Settings that set neither providers nor routes write
neither when saved.

Tests: `routes_left_out_go_to_the_embedded_provider`, `each_capability_goes_to_the_provider_its_route_names`

### R21 Providers and routes that cannot work are errors

The settings are refused, with what is wrong, when:

- a route names a provider `[providers]` does not have;
- a provider other than the embedded one has no address;
- a route has no `model` and its provider does not name its own;
- `max_questions` is 0, or `min_probability` is not from 0 to 1;
- one model name is asked of two providers (other clients' requests are forwarded by model
  name).

The `mode`, `remote_url` and `remote_key` of earlier settings files are unknown fields.

Tests: `providers_and_routes_that_cannot_work_are_errors`
