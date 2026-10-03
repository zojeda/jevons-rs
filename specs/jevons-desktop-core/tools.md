# jevons-desktop-core: tools

[Back to jevons-desktop-core](spec.md)

The tool host runs what flow nodes and tool loops call: the built-in tools (`command`, `http`, `open`)
and MCP servers registered in the desktop settings, and the library's automations as
`script:<name>`. It asks before a call runs, keeps each tool to the nodes its `allow` names, and in
a dry run records a call instead of making it. Agents run their tool-calling loop on adk-rust,
over the jevons API, through `JevonsLlm`. The node files (`tool.toml`, `loop.toml`, `run.toml`)
and the checks made when they load are in [flows](flows.md). A machine's `tools` list is in
[machines](machines.md).

## Requirements

### R1 Tools come from the settings

`[tools.<name>]` in `jevons-desktop.toml` registers a built-in tool, and `[mcp.<name>]` an MCP
server. Nothing in the flows folder registers a tool. A `kind` other than `command`, `http` or
`open` fails to parse. A built-in tool without its `program` (`command`) or `url` (`http`,
`open`), or with a `{placeholder}` that names none of its `arguments`, stays out of the host, and
the log says why. `${env:NAME}` is not a placeholder.

Tests: `tools_and_mcp_servers_parse_and_check_their_placeholders`

### R2 Placeholders and escaping

A call fills `{argument}` with the argument's value and `${env:NAME}` with that environment
variable, or nothing when it is unset. An `http` address gets the value percent-encoded (all but
unreserved characters), an `http` body gets it JSON-escaped, and every other field gets it as
written. A placeholder with no value stays as written. A built-in tool's arguments are strings,
and all of them are required.

Tests: `placeholders_fill_with_the_escaping_each_field_needs`, `builtin_tools_resolve_by_name_ask_by_default_and_respect_allow`

### R3 `command` runs a program without a shell

The filled `program` runs with each element of `args` as one argument, never through a shell, so
`;` or `$(...)` in a value stays text. Its environment holds `PATH`, `SYSTEMROOT`, `SYSTEMDRIVE`,
`WINDIR`, `TEMP`, `TMP`, `HOME`, `USERPROFILE`, `LANG` and the names in `env`, when they are set,
and nothing else. `stdin` goes to its standard input and `cwd` sets its folder. On Windows no
console window opens. The result is `{"output": <standard output>}`. A failing exit status fails
the call with the status and the first 2,000 characters of standard error.

Tests: `a_command_runs_without_a_shell_and_reports_its_output`

### R4 Tools give up at their time limit

A built-in tool call that runs past `timeout_s` (20 by default, at least 1) fails, and a program
still running is killed.

Tests: none yet

### R5 `http` sends one request

The method is `method`, or `POST`. Headers are filled as written. A body that parses as JSON once
filled is sent as JSON, and any other body as text. A success answers
`{"status": <code>, "body": <JSON or text>}`. An error status fails the call with the status and
the first 2,000 characters of the body.

Tests: none yet

### R6 `open` opens an address or file

The filled `url` (percent-encoded when it holds `://`) opens with the system's default application:
`explorer` on Windows, `open` on macOS, `xdg-open` elsewhere. The result is
`{"opened": <target>}`.

Tests: none yet

### R7 Output is capped

A `command`'s standard output and an `http` body keep their first 20,000 characters, followed by
`…`.

Tests: none yet

### R8 Results read as text

The bubble and the next node read a result as text: an MCP result's text parts joined by
newlines, a string as it is, an object with one field as that field's value, and anything else as
pretty-printed JSON.

Tests: `results_read_as_text`

### R9 Tools are named as flow files name them

A node names a built-in tool by its name, an MCP tool as `server:tool`, every tool of a server as
`server:*`, an automation as `script:<name>` and every automation as `script:*`. The model knows
them as `server__tool` and `script__<name>`, since its tool names cannot hold a colon. A name that
matches no tool, server, server tool or automation fails to resolve with a message that names it.

Tests: `an_mcp_server_starts_lists_its_tools_and_answers_calls`, `automations_are_script_tools_that_ask_and_respect_allow`

### R10 Every tool asks first unless the settings say otherwise

Built-in tools, MCP servers and automations ask before each call. `confirm = false` on a built-in
tool or a server turns that off, a server's `unconfirmed` names tools that run without asking,
and `automation.unconfirmed` names automations that do. `confirm = true` in a `tool.toml` asks
even for a tool the settings let run. Nothing outside the settings turns asking off.

Tests: `builtin_tools_resolve_by_name_ask_by_default_and_respect_allow`, `an_mcp_server_starts_lists_its_tools_and_answers_calls`, `automations_are_script_tools_that_ask_and_respect_allow`

### R11 `allow` limits which nodes call a tool

`allow` on a built-in tool or a server, and `automation.allow.<name>`, hold globs on flow node
paths, with leading and trailing `/` ignored. A node that matches none of them cannot resolve the
tool, and the call fails with `<tool> does not allow the node <node>`. An empty `allow` allows
every node.

Tests: `builtin_tools_resolve_by_name_ask_by_default_and_respect_allow`, `automations_are_script_tools_that_ask_and_respect_allow`

### R12 MCP servers start, list their tools and answer calls

A server's `command` is a program and its arguments, spoken to over standard input and output,
with `env` (whose values may use `${env:NAME}`) and `cwd`. The host starts every server when asked
and lists its tools, and reports one problem per server that cannot start or list. A server not
started yet starts on its first call. A server that failed stays failed. The catalog that flow
files are checked against names a server's tools only once it has listed them.

Tests: `an_mcp_server_starts_lists_its_tools_and_answers_calls`

### R13 `TOOLS.md` lists what the settings register

`TOOLS.md` names every registered tool as flow files call it, with its description, whether it
asks in the bubble, and each argument with its type and whether it is required. An automation not
approved yet says it runs once approved. A server that has not listed its tools gets a
`server:*` entry that says why. With nothing registered, it says so.

Tests: `builtin_tools_resolve_by_name_ask_by_default_and_respect_allow`, `automations_are_script_tools_that_ask_and_respect_allow`, `an_mcp_server_starts_lists_its_tools_and_answers_calls`

### R14 A call that asks runs only when approved in time

A call that asks goes to the confirmer with the tool's name and arguments, and the take waits for
the answer. The call runs only on a yes within 60 seconds. A no, no answer in time, or no
confirmer denies it. A denied tool node ends the take with `<tool> did not run: it was not
confirmed`, and the trace records the call with `confirmed: false`. In a machine, the denial is
the `denied` event ([machines](machines.md)).

Tests: `a_call_runs_only_when_approved_in_time`, `an_unconfirmed_tool_call_does_not_run`, `a_declined_tool_call_takes_the_denied_transition`

### R15 A dry run records calls instead of making them

A host in dry-run mode resolves, lists and checks tools as usual, and every call returns
`{"dry_run": true, "would_call": <reference>, "arguments": <arguments>}` without running.
Headless takes use it with no confirmer, so a call that asks is denied and one that does not
returns the record.

Tests: `builtin_tools_resolve_by_name_ask_by_default_and_respect_allow`, `a_tool_node_fills_its_arguments_asks_and_answers_with_the_result`

### R16 Tool nodes fill their arguments

Each argument of a tool node comes from one source. `value` is text with placeholders. `generate`
has the generative model write the value, one request per argument, within the generation time
limit. `choose` picks one of its labels and `noul` answers yes or no; all of a node's `choose` and
`noul` arguments go in one System One request. Without an answer, `choose` takes its first label
and `noul` is false. Text is read as the argument's `type`, or the tool's schema type: an integer
or number that does not parse fails the call, and a boolean is true for `true`, `yes` or `1`.

Tests: `a_tool_node_fills_its_arguments_asks_and_answers_with_the_result`

### R17 Tool nodes send their result where `output` says

The result goes to the bubble by default, or to the target application, the clipboard, nowhere,
or (with `next`) into the node's single branch as `{result}`. A failed call fails the take with
the tool's name and error. The trace keeps each call's node, tool, arguments, the first 600
characters of its result, and whether the user approved it.

Tests: `a_tool_node_fills_its_arguments_asks_and_answers_with_the_result`

### R18 Agents run a bounded tool-calling loop

A `loop.toml` node runs an adk-rust agent over the tools it names, for at most `max_steps`
model turns (4 by default). When the take has a context investigator, the agent can also call
`investigate`, which takes a `question` and answers from the screen. Its instruction is the
instructions gathered from the root, plus one telling it to call tools one at a time and then
answer in plain text. Its input is the node's `prompt`, or the take's context, investigations and
words. Its answer is the last text the model wrote without calling a tool.

Tests: `an_agent_calls_a_tool_reads_its_result_and_answers`, `an_agent_node_calls_its_tools_and_its_answer_goes_to_the_bubble`

### R19 An agent's calls ask first and show in the bubble

An agent's tools that ask go through the confirmer. A denied call does not run, and the agent
goes on without its result. Each call is a stage in the bubble. With `output = "bubble"` (the
default), the answer streams into the bubble as the model writes it. The trace records each call
under the name flow files use, with its arguments and the first 600 characters of its result.

Tests: `a_denied_confirmation_keeps_the_tool_from_running`, `an_agent_node_calls_its_tools_and_its_answer_goes_to_the_bubble`

### R20 `JevonsLlm` speaks Chat Completions

Each model turn of an agent is one streamed `POST /v1/chat/completions`. User and system text
become messages, the model's calls an assistant message with `tool_calls`, and results `tool`
messages that answer their call's id. The tools go sorted by name, with their descriptions and
parameters. The turn's output token limit becomes `max_completion_tokens`, a structured answer a
JSON Schema `response_format`, and a thought budget the smallest `reasoning_effort` that covers
it. Text streams out as it arrives, and a call arrives whole and ends the turn.

Tests: `adk_contents_become_chat_turns_with_calls_and_results`, `a_streamed_tool_call_is_joined_and_its_arguments_parsed`

### R21 Automations are `script:<name>` tools

Each automation in the library is the tool `script:<name>`, described by its manifest, with the
manifest's arguments as its schema (required unless they have a default). A call runs the
approved version and answers with the script's result, or `{"done": true}` when it has none. A
version not approved fails with `not_approved` and acts on nothing. A failed run fails the call
with its message and `script.rhai:<line>:<column>`.

Tests: `automations_are_script_tools_that_ask_and_respect_allow`, `only_approved_versions_run_and_tools_carry_their_arguments`

### R22 Run nodes run an approved automation with arguments from the words

A `run.toml` node runs one of the approved automations its `automations` allows (all when empty
or `*`). With none in the library, it fails and says to record one. With none approved, it fails
and names those that wait. One approved automation runs as it is; among several, the decision
model chooses by their descriptions, and the first one runs when it gives no answer. Its
arguments come from what the user said: a boolean as a yes-or-no question, anything else written
by the generative model. It runs as the tool `script:<name>`, asking first like any call, and its
answer goes to `output` (the bubble by default).

Tests: `a_run_node_fills_an_automation_s_arguments_from_the_words_and_runs_it`
