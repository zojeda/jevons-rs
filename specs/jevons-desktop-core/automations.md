# jevons-desktop-core: automations

[Back to jevons-desktop-core](spec.md)

An automation is a Rhai script that reads an application's interface with XPath and acts on it,
such as posting a message to a Slack channel. The library is a folder of them, one per subfolder.
Each one is checked before it runs and dry-run against the demonstrations it was written from, and
it runs for real only in the version the user approved. The author writes one from a recording.
Recording is in [recording](recording.md), `script:<name>` tools and `run.toml` nodes in
[tools](../jevons-desktop-server/tools.md), and the tray, hotkeys and the approval prompt in jevons-desktop.

## Requirements

### R1 The library is a folder of automations

Each subfolder of the library holds one automation, named by the folder: `automation.toml`,
`script.rhai` and the `fixtures/` it replays. A name uses lowercase letters, digits, `-` and `_`,
starts with a letter or digit, and has at most 64 characters. Folders whose names start with `_`
or `.` are skipped. A folder with another name, or with a missing or wrong file, is reported with
its name and the problem, and the others load.

Tests: `a_library_reads_its_folders_and_reports_the_bad_ones`

### R2 The manifest says what an automation may touch

`automation.toml` holds `description`, `apps` (process-name globs), `timeout_s` (1 to 300, 30 by
default), `[args.<name>]` (a `description`, a `type` of `string`, `integer`, `number` or
`boolean`, and an optional `default`), `returns` (the answer's shape, as in an `[investigate]`)
and `[[fixtures]]` (a `recording` inside the folder, and its `args`). Unknown fields are errors.
An empty `description` or `apps`, a bad glob, an argument name that is not lowercase letters,
digits and `_`, a default or fixture argument of the wrong type, and a `recording` path that
leaves the folder are errors too. Each error names its field.

Tests: `manifest_mistakes_are_reported_by_field`, `a_manifest_reads_its_arguments_and_fills_calls`

### R3 A call's arguments are checked and typed

An argument the manifest does not declare fails the call and names the ones it takes. A required
argument left out fails the call, and a default fills an optional one. Integers and numbers may
come as text, and booleans as `true`, `false`, `yes` or `no`. A string argument takes a number or
a boolean as its text. The arguments' JSON Schema lists every argument with its type and requires
those without a default.

Tests: `a_manifest_reads_its_arguments_and_fills_calls`

### R4 A version is the hash of both files

An automation's version is `sha256:` and the SHA-256 of its manifest and script, with `\r\n` read
as `\n`. Changing either file changes the version.

Tests: `the_version_ignores_line_endings_and_changes_with_either_file`

### R5 Only the approved version runs

An approval pins a version in the settings' `[automation.approved]`. A run of an automation whose
version is not pinned fails with `not_approved` before it reads or acts on anything, so an edited
script needs approving again. Nothing in the library folder can pin a version. Dry runs need no
approval.

Tests: `only_approved_versions_run_and_tools_carry_their_arguments`, `a_library_reads_its_folders_and_reports_the_bad_ones`

### R6 Scripts run in a sandbox

A script has no file, network, process or module access. `eval`, `Fn`, `call`, `curry`, `import`
and `export` are disabled, every variable must be declared, and a name cannot be declared twice in
one scope. A run stops with `limit` past 1,000,000 operations, 32 call levels, an expression depth
of 64 (32 inside functions), a string of 1 MiB, 10,000 array or map entries, 1,000 variables or
64 functions.

Tests: `scripts_catch_typed_errors_and_limits_stop_them`, `mistakes_are_found_before_running_with_their_line_and_column`

### R7 `API.md` is the script API

Scripts call the queries (`find`, `find_all`, `try_find`, `exists`, `text`, `wait_for`,
`wait_gone`, `window`, `windows`), the element actions (`invoke`, `click`, `focus`, `set_value`,
`type_text`, `toggle`, `select`, `expand`, `collapse`, `scroll_into_view`), `press` and
`type_text` for the window in front, `activate` for a window, and `step`, `log`, `sleep`,
`confirm` and `fail`. `API.md` documents every function the engine registers, and no function it
does not.

Tests: `api_md_documents_every_function_scripts_can_call_and_no_other`

### R8 Queries read the automation's own windows

A query evaluates its expression in the front window of the automation's applications, or below
the window or element it is called on. Its `$variables` come from the map passed with it, then
from the arguments. `find` fails with `not_found` when nothing matches and `ambiguous` when
several elements do. `try_find` gives `()` when nothing matches. `wait_for` and `wait_gone` look
again every 100 ms and fail with `timeout` at their limit. A query with no window of its
applications open fails with `not_found`.

Tests: `scripts_catch_typed_errors_and_limits_stop_them`, `a_script_replays_its_demonstration_and_answers_in_its_shape`

### R9 Failures are typed and say where

A failure is a map with `kind` and `message`, plus `xpath` and `count` for a query. The kinds are
`not_found`, `ambiguous`, `timeout`, `denied`, `action_failed`, `cancelled`, `invalid`,
`platform`, `script` and `limit`. `try`/`catch` reads them, and `fail(kind, message)` raises one.
A run that ends with a failure reports its kind, message, line and column, and the expression it
was about.

Tests: `scripts_catch_typed_errors_and_limits_stop_them`, `wrong_arguments_and_wrong_steps_fail_with_where_and_why`

### R10 Actions pass the hands' checks

An automation acts only on elements of its `apps`, and brings forward only their windows. It never
types into a password field, and a password field's `value` reads empty. A disabled element takes
no action but focus and scrolling. Keys and typed text go to the window in front only when that
window belongs to one of its applications, and otherwise fail with `denied`, so they never land in
another application. Every action is recorded with how it was done, or why not.

Tests: `actions_happen_only_in_the_automation_s_applications`, `passwords_and_disabled_elements_are_refused_and_the_rest_recorded`

### R11 Static checks find mistakes before a run

Before anything runs, the script must compile with `args` known and every variable declared, and
every function it calls must exist: the script API, Rhai's standard library, or its own. Every
XPath expression written as text must parse, and its `$variables` must be arguments or keys of
the map passed with it. Every key chord must parse, and `window()` must name one of its `apps`.
Each problem reads `script.rhai:<line>:<column>: <message>`.

Tests: `mistakes_are_found_before_running_with_their_line_and_column`

### R12 The summary says what a script does

The checks report, for whoever approves the script, its applications, the element actions it
takes, the chords it presses, whether it types into the window in front, how many queries it
makes, and whether it asks with `confirm()`.

Tests: `the_summary_says_what_the_script_does`

### R13 A dry run replays a demonstration step for step

In a dry run, queries read the interface recorded before the step being replayed. An action that
matches that step moves the replay to the next step's interface. Any activation (`invoke`,
`click`, `select`, `toggle`, `expand`, `collapse`) of the same element matches, and so does the
same text however it is entered; focusing and scrolling need no step. Any other action fails with
`action_failed`, naming the step expected and where it is (`step 1 of 3`). A script that ends
before the last step fails with `incomplete`. `confirm()` answers yes.

Tests: `a_script_replays_its_demonstration_and_answers_in_its_shape`, `wrong_arguments_and_wrong_steps_fail_with_where_and_why`, `a_replay_moves_on_with_each_matching_action_and_explains_the_others`

### R14 The checks gate approval

An automation's checks are the static checks and then, when those pass, a dry run of each fixture
with its `args`. It passes when there is no problem and every fixture replays. The example
library in `examples/desktop/automations` passes.

Tests: `the_example_library_passes_its_checks_and_replays_its_fixtures`, `a_draft_takes_the_values_the_user_said_as_arguments_and_replays`

### R15 A run stops at its deadline or when cancelled

A run stops with `timeout` at `timeout_s` and with `cancelled` when cancelled, between operations
and inside every wait. `step()` labels and `log()`, `print` and `debug` lines go to the run's
trace, never to the log file. The answer is the script's last expression, fitted to `returns`.

Tests: `a_cancelled_or_late_run_stops_inside_its_waits`, `a_script_replays_its_demonstration_and_answers_in_its_shape`

### R16 Step by step asks before every action

A step-by-step run asks before each action, naming it (`invoke TreeItem "random" #C03RANDOM33`).
A no stops the run with `cancelled`, and nothing after it acts.

Tests: `step_by_step_asks_before_each_action_and_no_stops_it`

### R17 `confirm()` asks the user

`confirm(message)` asks the user yes or no and returns the answer. With no one to ask, the
answer is no.

Tests: none yet

### R18 A failed run keeps the interface

A live run that fails with any kind but `invalid` or `cancelled` writes `failures/<time>.json`
into the automation's folder: the run's trace and the interface of its applications' windows (40
levels deep, at most 5,000 elements). The newest ten are kept.

Tests: none yet

### R19 jevons writes the library's guides

The library gets `AGENTS.md`, `API.md`, `_schemas/automation.schema.json` and a `.taplo.toml`
that maps the schema for editors. jevons refreshes the guides it wrote and leaves an edited one
as it is, with a note. A `.taplo.toml` it did not write stays as it is. A library that is up to
date gets no writes.

Tests: `a_library_folder_gets_its_guides_and_schema_and_keeps_edited_ones`

### R20 The author plans with the model or drafts without it

A plan names the automation, describes it, binds each argument to a value the recording used, and
gives each step a label and one of its candidate expressions. With a model, the plan is one chat
request whose JSON Schema offers each step's candidates as its only choices; a name is made a
valid one, and an answer outside the choices falls back to the draft's. Without a model, the
draft plans from the recording alone, and the values the user also said become the arguments.

Tests: `a_models_plan_is_read_within_the_choices_and_mistakes_fall_back`, `a_draft_takes_the_values_the_user_said_as_arguments_and_replays`, `without_a_model_the_draft_is_written_and_checked`

### R21 A plan compiles to the same files every time

A plan compiles into `automation.toml` and `script.rhai`. The script labels each step, waits for
each element, and uses an argument wherever the recording had its value. The recording becomes the
first fixture, with those values as its `args`, so the result replays what was done.

Tests: `a_draft_takes_the_values_the_user_said_as_arguments_and_replays`, `a_models_plan_is_read_within_the_choices_and_mistakes_fall_back`

### R22 The author writes and checks a new version

The author writes the automation under its name, or `<name>-2`, `<name>-3` and on when the name
is taken. Replacing writes over the named automation and keeps its older fixtures: a new version
to approve again. The result goes through every check. When the model's plan fails them, the
draft takes its place.

Tests: `a_draft_takes_the_values_the_user_said_as_arguments_and_replays`, `a_models_plan_that_replays_is_the_one_written`, `without_a_model_the_draft_is_written_and_checked`, `the_example_recording_loads_and_its_draft_replays`

### R23 Automations are off until the settings turn them on

Automations are coming soon. While `[automation] enabled` is not `true`, the host lists no
automation, so none is offered as a tool, and a run fails with `invalid` and a message that says
so, before it reads or acts on anything, approved or not. The library is still loaded: checks,
dry runs and the author work as they do.

Tests: `automations_are_off_until_the_settings_turn_them_on`
