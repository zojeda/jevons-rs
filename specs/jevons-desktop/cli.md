# jevons-desktop: cli

[Back to jevons-desktop](spec.md)

Without flags, `jevons-desktop` starts the tray app. Its flags run one thing headless and exit:
takes from audio or text, checks of a flows folder or an automations library, XPath queries,
automation dry runs and runs, writing an automation from a recording, a settings reset and
history clears. Headless commands print their result on standard output, log to standard error,
and exit with an error when what they ran failed.

## Requirements

### R1 `--config` names the settings file

`--config <file>` is the settings file every mode uses. Without it, the app uses the platform's
(`jevons-desktop.toml` in the configuration folder).

Tests: none yet

### R2 Headless commands log to standard error

Any flag below makes the run headless: its log goes to standard error, and standard output holds
only its result. Without one, the app runs in the tray and logs to its file.

Tests: `flow_commands_take_an_optional_folder`, `clearing_takes_kinds_of_history_or_all_of_it`

### R3 `--replay` runs one take from an audio file

`--replay <audio>` plays the file as the microphone, at its own pace, through one take, and prints
the trace as JSON. It cannot be combined with `--transcript`.

Tests: none yet

### R4 `--transcript` runs takes from text

`--transcript <text>` runs a take from that text as if it had been said. Repeated, it runs one
take per text in order, sharing the machines, so a task waits between them. One take prints its
trace as JSON, and several print an array of traces.

Tests: none yet

### R5 Headless takes run as the app does, without acting

`--replay` and `--transcript` use the settings' runtime (embedded or remote) and flow tree, and
wait for the runtime to be ready; without one they fail with its status. The context is the
snapshot in `--context <json>`, or empty. `--flow <branch>` starts the take at that branch. `--tree
<json>` answers extracts and investigations from a recorded interface instead of the live one.
The tools run as a dry run with no one to confirm them: MCP servers are started and listed, and
nothing runs. A flow tree with problems prints them and fails. Progress (the words heard, each
stage with ✓ or ✗, the machines' states) goes to standard error. The command fails with the first
take's error.

Tests: none yet

### R6 `--deliver` types the result

With `--deliver`, a headless take delivers its text into the focused application, with the usual
checks. Without it, the text is only printed.

Tests: none yet

### R7 `--check-flows` checks a flows folder

`--check-flows [DIR]` loads the folder (by default the settings' one) against the tools the
settings register and the automations library, with MCP tools checked by server since their
servers are not started. It prints every problem with its file and line on standard error and
fails when there is one; otherwise it prints how many nodes and leaves the tree has.

Tests: `flow_commands_take_an_optional_folder`

### R8 `--init-flows` writes the built-in tree

`--init-flows [DIR]` writes the built-in flow tree into a folder that has none (by default the
settings' one), refreshes `AGENTS.md`, the schemas and `.taplo.toml`, prints each file written or
removed and each note, commits them in jevons' settings repository, and then checks the folder as
`--check-flows` does.

Tests: `flow_commands_take_an_optional_folder`

### R9 `--xpath` prints what an expression selects

`--xpath <expression>` evaluates the expression in the window in front, reading that window only.
`--app <glob>` reads the first window whose process name matches instead, and `--tree <file>` a
recorded interface. It prints one line per element selected (at most 100), or the value, and
reports on standard error the window read, how many matches, the milliseconds taken and how many
elements it read. An expression that does not parse prints with a caret under the column of the
mistake, and fails. With no window to read, it fails.

Tests: none yet

### R10 `--check-automations` checks a library

`--check-automations [DIR]` checks every automation of the library (by default the settings'
one): the script checks, then each fixture's dry run. It prints, per automation, whether it passes
and whether this version is approved, how many steps each fixture replays, and what it does in
which applications. Problems go to standard error, and any problem fails the command.

Tests: none yet

### R11 `--dry-run` and `--run` run one automation

`--dry-run <name>` replays the automation against its first fixture, or the demonstration in
`--recording <json>`, with the arguments in `--args <json>` or the fixture's. `--run <name>` runs
an approved automation on the live interface, printing each step on standard error. Both print the
run's trace as JSON and fail with its error, its line and its column. `--library <dir>` picks the
library. `--args` that is not a JSON object fails. The two cannot be combined.

Tests: none yet

### R12 `--author` writes an automation from a recording

`--author <recording>` writes an automation into the library from a saved recording's folder,
drafted from the recording alone, commits it in jevons' settings repository, and prints its
checks. `--replace <name>` makes it a new version of that automation. It fails when the result
does not pass its checks.

Tests: none yet

### R13 `--reset-settings` puts the defaults back

`--reset-settings` resets the settings folder as the tray does (jevons-desktop-core's
[settings](../jevons-desktop-core/settings.md)), before loading the settings, so it also mends a
file that does not load. It prints the entries removed, the folder, whether it was committed, and
each note. Quit the tray app first.

Tests: none yet

### R14 `--clear` clears kinds of history

`--clear <what>` takes `logs`, `traces`, `trees`, `recordings` or `all`, several separated by
commas, and clears each kind once. It prints what each clear did and where. Any other kind, such
as `models`, is refused. It fails when a file could not be removed.

Tests: `clearing_takes_kinds_of_history_or_all_of_it`
