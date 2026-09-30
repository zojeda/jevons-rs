# Automations: scripts jevons runs in your applications

Each folder here is one automation: a script that reads an application's interface with XPath
and acts on it, such as posting a message to a Slack channel. The user runs it from the tray, a
hotkey, a flow branch ("post to random that lunch is ready") or an agent.

```
slack-post/
  automation.toml    what it does, its applications, arguments and answer
  script.rhai        the script (API.md lists what it can call)
  fixtures/          recorded demonstrations it must replay
```

The folder name is the automation's name. Use lowercase letters, digits, `-` and `_`: the
decision model answers with it. Folders starting with `_` or `.` are ignored.

## `automation.toml`

```toml
description = "Posts a message to a Slack channel"   # for the tray and the decision model
apps = ["slack.exe"]            # the only applications it may read and act on
timeout_s = 30                  # at most 300
returns = { posted = "boolean" }  # the answer's shape, as in a flow's [investigate]

[args.channel]
description = "The channel's name, without #"   # for whoever fills it: the user or a model
[args.message]
description = "What to post"
[args.urgent]
description = "Whether to mark it urgent"
type = "boolean"                # string (default), integer, number or boolean
default = false                 # optional arguments have a default

[[fixtures]]
recording = "fixtures/post.json"      # a demonstration, relative to this folder
args = { channel = "random", message = "lunch is ready" }
```

Unknown fields are errors. `_schemas/automation.schema.json` describes the file for editors.

## `script.rhai`

The script is Rhai, with the API in `API.md`: queries such as `find("//TreeItem[...]")`, element
actions such as `.invoke()` and `.type_text(text)`, and `press("enter")`, `step("…")` and
`wait_for(xpath, ms)`. Write it to be robust:

- **Classes and ids over names.** Names are in the user's language, so prefer
  `has-class(@class, '…')` and `@automation_id` to `@name`, except for the values the arguments
  carry (`@name = $channel`).
- **Wait, don't sleep.** Wait for what an action brings (`wait_for`) instead of sleeping.
- **Label the steps.** Call `step()` before each part, so the user sees what happens.
- **Fail with a reason.** Use `fail("not_found", "…")` with a reason the user understands,
  instead of guessing.

## Fixtures and checks

A fixture is a demonstration the user recorded: the interface before each step, and what they
did (`fixtures/*.json`). A dry run replays it:
- queries read each step's interface;
- each action must be the step demonstrated, on the same element with the same text;
- the script must do every step.

Check your work before you finish:

```sh
jevons-desktop --check-automations <this folder>     # every automation: script checks, then each fixture
jevons-desktop --dry-run slack-post --args '{"channel": "random", "message": "lunch is ready"}'
```

Before anything runs, the checks cover:
- **Syntax:** it compiles, and every variable is declared.
- **Calls:** every function it calls exists.
- **Written-out values:**
  - XPath expressions must parse;
  - their `$variables` must be arguments or be passed with the query;
  - key chords must be valid;
  - `window()` must name one of its `apps`.

Errors give the file, line and column.

## Running and approval

An automation runs only when the user has approved its current version in jevons, which pins
the hash of `automation.toml` and `script.rhai` in the desktop settings. Changing either file
makes it a new version: it keeps working in dry runs, and runs for real only once the user
approves it again. Nothing in this folder can approve an automation. Runs also ask in the bubble
before they start, unless the settings list the automation as unconfirmed.

A run fails, with its kind, line and column in the trace, when:
- nothing matches an expression that had to match;
- an element cannot do the action asked of it;
- the window in front is not one of the automation's applications;
- the time runs out.

Failed runs keep the window's interface in `failures/`, to fix the script against.
