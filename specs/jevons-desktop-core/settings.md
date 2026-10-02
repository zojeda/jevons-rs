# jevons-desktop-core: settings

[Back to jevons-desktop-core](spec.md)

The settings folder holds `jevons-desktop.toml` and, next to it unless the settings move them, the
flow tree in `flows/` and the automations library in `automations/`. jevons fills in whatever the
folder lacks, keeps it in a git repository of its own, commits what it writes there, and can put
the defaults back with the earlier settings kept in the history. The history clears remove the
logs, traces, recorded interfaces and recordings jevons keeps under `~/jevons`. The tools and MCP
servers the file registers are in [tools](tools.md), and the model selections in
[client](client.md). The tray and the Settings tab that drive all this are in jevons-desktop.

## Requirements

### R1 Where the settings live

The settings file is `--config`, or `jevons-desktop.toml` in the platform configuration folder
(`%APPDATA%\jevons\config` on Windows, `~/.config/jevons` on Linux). The flow tree is `flows_dir`,
or `flows/` next to the file. The automations library is `[automation] dir`, or `automations/`
next to the file. `models.toml` sits next to the file. What jevons keeps of past use goes under
`~/jevons`: `logs/`, `traces/`, `trees/`, `recordings/` (or `[automation] recordings_dir`), and
`models/` (or `[models] folder`).

Tests: `approvals_are_written_alone_and_saving_the_settings_keeps_them`

### R2 A missing file means defaults, and unknown fields are errors

A settings file that does not exist loads as the defaults. A field the file format does not have,
in any section, fails the load with the file's path and the parser's message.

Tests: `a_missing_file_gives_defaults_and_unknown_fields_are_errors`, `the_example_settings_file_parses`

### R3 The file's sections and defaults

| Section | Fields and defaults |
| --- | --- |
| `[server]` | `mode` (`embedded` or `remote`, `embedded`), `expose` (false), `bind` (`127.0.0.1`), `port` (8080), `api_key` (none), `remote_url` (`http://127.0.0.1:8080`), `remote_key` (none) |
| `[models]` | `folder`, `runtime_config`, `generative`, `decision` and `speech` (each a `path`, an optional `mmproj` and the `catalog` entry it came from), `realtime` (true) |
| `[dictation]` | `hotkey` (`Ctrl+Alt+Space`), `hotkey_mode` (`hold`), `live_hotkey` (`F9`), `live_hotkey_mode` (`hold`), `live_feedback` (true), `inspector_hotkey` (none), `branch_hotkeys` (none), `microphone` (the default device), `language` (detected), `decide` (true), `max_output_tokens` (1024) |
| `[privacy]` | `max_context_chars` (2000), `read_clipboard` (false), `read_other_windows` (false), `readable_apps` (none), `log_api` (false) |
| `[automation]` | `dir`, `recordings_dir`, `record_hotkey` (none), `hotkeys` (none), `approved` (none), `unconfirmed` (none), `allow` (none), `author_model` (the generative model) |
| top level | `flows_dir`, `[tools.<name>]`, `[mcp.<name>]` |

The example file, `jevons-desktop.example.toml`, parses.

Tests: `the_example_settings_file_parses`, `saved_settings_load_back_unchanged`

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

On every start, jevons writes the settings file when there is none, the built-in flow tree and its
guides ([flows](flows.md)), and the automations library's guides ([automations](automations.md)).
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
settings file, `flows/`, `automations/` and `.git`. Any other folder is left as it is, and the
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
