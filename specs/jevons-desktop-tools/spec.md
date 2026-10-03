# jevons-desktop-tools

## Purpose

`jevons-desktop-tools` is the tools jevons runs from its settings: built-in tools (a program, an
HTTP request, an address opened) and MCP servers. A tool runs on the side whose settings file
registers it, so the desktop server and its client both use this crate, each over its own file.

## Scope

It owns the settings' tool types (`ToolConfig`, `ToolKind`, `McpConfig`) and their checks, the
filling of `{argument}` and `${env:NAME}`, the three built-in runners, MCP servers started over
standard input and output, and `ToolSet`: one side's tools, what it knows of them, and each tool
a reference names, ready to call.

It leaves to its users whether a call may run: the flow node's `allow`, the user's yes and dry
runs are the server's for its own tools ([tools](../jevons-desktop-server/tools.md)) and the
client's for its own ([desk](../jevons-desktop-core/desk.md)).

## Requirements

### R1 Tools come from a settings file

`[tools.<name>]` registers a built-in tool, and `[mcp.<name>]` an MCP server. A `kind` other than
`command`, `http` or `open` fails to parse. A built-in tool without its `program` (`command`) or
`url` (`http`, `open`), or with a `{placeholder}` that names none of its `arguments`, stays out of
the set, and the log says why. `${env:NAME}` is not a placeholder, and is that environment
variable, or nothing when it is unset.

Tests: `tools_and_mcp_servers_parse_and_check_their_placeholders`, `environment_variables_fill_in_and_missing_ones_are_empty`, `a_set_knows_its_tools_and_resolves_them_by_reference`

### R2 Placeholders and escaping

A call fills `{argument}` with the argument's value and `${env:NAME}` with that environment
variable, or nothing when it is unset. An `http` address gets the value percent-encoded (all but
unreserved characters), an `http` body gets it JSON-escaped, and every other field gets it as
written. A placeholder with no value stays as written. A built-in tool's arguments are strings,
and all of them are required.

Tests: `placeholders_fill_with_the_escaping_each_field_needs`

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

### R8 MCP servers start, list their tools and answer calls

A server's `command` is a program and its arguments, spoken to over standard input and output,
with `env` (whose values may use `${env:NAME}`) and `cwd`. A set starts every server when asked
and lists its tools, and reports one problem per server that cannot start or list. A server not
started yet starts on its first call. A server that failed stays failed. A set knows a server's tools only once it has listed them.

Tests: `an_mcp_server_starts_lists_its_tools_and_answers_calls`

### R9 A set knows its tools and resolves them by reference

A set knows each built-in tool, and each tool of the MCP servers that have listed theirs, with
its description, its arguments' schema, whether it asks first (a built-in tool's `confirm`; a
server's `confirm` unless the tool is in its `unconfirmed`) and its `allow`. A reference resolves
to the tools it names: a built-in tool's name, `server:tool`, or `server:*` for all of a
server's, each with the name the model calls it by (`server__tool`, since a tool's name cannot
hold a colon). A name nothing registers fails with a message that names it.

Tests: `a_set_knows_its_tools_and_resolves_them_by_reference`, `an_mcp_server_starts_lists_its_tools_and_answers_calls`

### R10 `allow` is globs on flow nodes

`allowed` says whether a flow node may call a tool: one of the tool's `allow` globs matches the
node's path, with leading and trailing `/` ignored. A tool with no `allow` allows every node.

Tests: `a_set_knows_its_tools_and_resolves_them_by_reference`
