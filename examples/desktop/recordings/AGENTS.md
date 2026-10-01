# A recorded task, to turn into an automation

The user recorded this task in jevons and said what it is:

> {description}

Write an automation that does it again, taking as arguments what should change from one run to
the next. Put the automation into the library, `{library}`, as a folder of its own (see the
library's `AGENTS.md`), with this recording as its first fixture.

## What is here

- `recording.json`: the notes the user spoke, and the steps. Each step has:
  - `did`: what the user did;
  - `deed`: the action;
  - `candidates`: XPath expressions that find the element in the interface of that step, most
    robust first;
  - `said`: values of the step the user also said, which are likely arguments.
- `demonstration.json`: every step with the interface before it. It is the fixture a dry run
  replays; it holds text from the user's screen, so keep it on this machine.
- `API.md`: what a script can call.
- `draft/`: a first version jevons compiled from the recording alone. It replays the recording,
  but its arguments and step labels are guesses: start from it, then improve it.

## How to write it

1. **Name it.** Choose a short name (lowercase letters, digits, `-`) and create
   `{library}/<name>/`.
2. **Write the manifest.** `automation.toml` gets:
   - a `description` the decision model can choose it by;
   - `apps` (the `app` of the steps);
   - an `[args.<name>]` for each value that should change, such as the channel and the message;
   - `returns`, when the automation answers with something.
3. **Copy the fixture.** Copy `demonstration.json` to `{library}/<name>/fixtures/{recording}.json`,
   and add it as a fixture. Its `args` are the values of this recording, so it replays exactly
   what was done.
4. **Write the script.** In `script.rhai`, take one query and one action per step:
   - use each step's most robust candidate, with the values replaced by `$arguments`
     (`//TreeItem[.//Text[@name = $channel]]`);
   - wait for what an action brings with `wait_for`;
   - call `step()` before each part.
5. **Check it,** until both commands pass:

   ```sh
   jevons-desktop --check-automations {library}
   jevons-desktop --library {library} --dry-run <name>
   ```

The user approves the automation in jevons before it runs for real. You cannot approve it.
