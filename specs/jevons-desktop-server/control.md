# jevons-desktop-server: control

[Back to jevons-desktop-server](spec.md)

The control API: what outside agents (Claude Code, scripts) reach through the server. One table of
operations (`control::ops`) and two doors built from it: MCP over streamable HTTP at `/mcp`, and
REST at `/api/v1/<op>` described by `/api/openapi.json`. The server answers `status` itself and
passes every other operation to the desk ([core](../jevons-desktop-core/control.md)), which checks
it against the client's settings.

## Requirements

### R1 One table, two doors

Every operation (`status`, `windows`, `focused`, `outline`, `find`, `xpath`, `read`, `act`, `press`,
`type_text`, `activate`, `show`, `ask`, `bubble`) is one MCP tool and one REST route, with the same
input and output types. MCP tools carry their input schema whole (no references), a title, and
whether they only read. REST takes `POST` with the input as a JSON body, and `GET` too for an
operation without input. The OpenAPI 3.1 document names every operation, its input, its output and
its errors, and every reference in it resolves.

Tests: `every_operation_is_named_once_and_reaches_the_desk`, `each_operation_is_a_tool_with_a_whole_object_schema`, `the_document_names_every_operation_and_every_reference_resolves`, `mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name`

### R2 Only with the key, never from a browser

The control API is built only with a key (`TYPESAFE_API_KEY`, else `[server] api_key`); without
one it is not served. Every request, MCP or REST, the document included, needs it as
`Authorization: Bearer <key>`: none or a wrong one is 401. A request with an `Origin` header (a
browser's) is 403. Keys are compared whole.

Tests: `rest_needs_the_key_refuses_browsers_and_reaches_the_desk`, `mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name`, `keys_compare_whole`

### R3 The server carries the desk's calls

`status` answers the version, the agent's name and what the desk lets it do, or why nobody is at the
desk. Every other operation becomes the desk's `control` call, whoever is at the desk now (the
app's own desk, or a client over `--serve`).

Tests: `rest_needs_the_key_refuses_browsers_and_reaches_the_desk`, `mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name`

### R4 A refusal says its kind

Input that does not fit an operation is `invalid`, naming the operation; an unknown operation is
`not_found`. Over REST a refusal is `{"error": {"kind", "message"}}` with a status per kind
(`invalid` 400, `unauthorized` 401, `denied` 403, `not_found` 404, `busy` 409, `failed` 500,
`unavailable` 503, `timeout` 504). Over MCP it is a tool error whose text starts with the kind.

Tests: `bad_input_is_invalid_and_says_which_operation`, `rest_needs_the_key_refuses_browsers_and_reaches_the_desk`, `mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name`

### R5 The agent says its name

An MCP call comes under the client's name (`clientInfo`, from the request or the session), a REST
call under its `X-Jevons-Agent` header; a call without one comes under `agent`. Names are cut at 60
characters. The desk keeps approvals by this name.

Tests: `mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name`, `rest_needs_the_key_refuses_browsers_and_reaches_the_desk`, `a_caller_without_a_name_is_an_agent`

### R6 MCP clients of either revision

A client of the 2026-07-28 revision calls without a session, and the list of tools tells it how
long it holds and that only this user may keep it (`ttlMs`, `cacheScope = "private"`); one of an
earlier revision starts with `initialize`. Either reads the server's instructions (under 4 KB):
what to call first, how ids work, the XPath vocabulary, what `denied` and `busy` mean, and how to
use the bubble.

Tests: `mcp_lists_the_operations_and_calls_them_as_the_client_that_says_its_name`, `each_operation_is_a_tool_with_a_whole_object_schema`

### R7 `[control]` turns it on

`[control] enabled` in `jevons-server.toml` (off by default) serves it on the server's listener
(`[server] bind:port`), beside `/v1` when exposed and `/desktop` under `--serve`.

Tests: none yet
