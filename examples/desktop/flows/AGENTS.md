# Flow tree: how jevons routes a take

This folder tells jevons what to do each time the user presses a dictation hotkey and speaks.
Every folder is a node. A take starts at the root node (or where its hotkey says), and each
decision picks one branch folder until a leaf writes the text somewhere. Edit the files and save:
jevons reloads the tree at once and, when a file has a problem, keeps the last good tree and shows
the errors in its inspector.

Check your changes before you rely on them:

```sh
jevons-desktop --check-flows <this folder>
jevons-desktop --transcript "reply that I agree" --context snapshot.json
```

The second command runs one take headless and prints its trace, with every decision's
probabilities. Write `snapshot.json` by hand or copy one from the inspector's Context tab (Raw
JSON); `examples/desktop/*.json` in the jevons repository has two. Add `--tree tree.json` to read
extracts and investigations from a recorded interface (the Context tab's **Record tree**), and
try an extract's expression on its own with `jevons-desktop --xpath "<expression>" --tree tree.json`.

## Nodes

A node folder holds exactly one node file, whose name is its kind:

| File | What it does |
|---|---|
| `decide.toml` | Chooses one of its branches: the subfolders, or the folders named by `branches`. |
| `generate.toml` | A leaf: the language model writes text, which goes to `output`. |
| `transcript.toml` | A leaf: the words as recognized go to `output`, with no model. |
| `tool.toml` | Calls one registered tool with arguments filled here. |
| `agent.toml` | A tool-calling loop over registered tools, bounded by `max_steps`. |
| `run.toml` | Runs one of the user's approved automations, with its arguments filled from what they said. |

A folder may also hold `instructions.md`: prose added to every model call at and below that
folder, taken exactly as written. Folders whose name starts with `_` or `.` are not nodes: use
`_name` folders for branches that several decisions share. Branch folder names are what the
decision model answers with, so they use lowercase letters, digits, `-` and `_`.

Generated files, which jevons rewrites: `_schemas/` (a JSON Schema per node file, which
`.taplo.toml` maps for editors) and `TOOLS.md` (the registered tools and their arguments). jevons
also updates this `AGENTS.md` until someone edits it.

## Fields every node file may set

| Field | Meaning |
|---|---|
| `description` | When this branch applies. Required below a `decide.toml`, whose model chooses by these descriptions: write them for it. |
| `priority` | Below `select = "rules"`, the highest priority among passing guards wins. |
| `[when]` | The guard, checked with no model call. A branch whose guard fails is not a candidate. |
| `[prefer]` | Rules (the same as `[when]`) that choose this branch with no model call: when they pass, the parent's decision chooses only among its preferred branches. A branch whose `[prefer]` fails is still a candidate. |
| `instructions` | Text added after `instructions.md` for this node and below. Placeholders allowed. |
| `delivery` | How text reaches the application: `paste` (default), `type`, `set_value`, `clipboard`. |
| `max_output_tokens` | The most tokens a generation writes, for this node and below. |
| `think` | A thought budget in tokens before a decision or generation. |
| `[extract.<name>]` | An XPath expression read from the application's interface, with no model (below). |
| `[investigate.<name>]` | A question for the built-in context investigator (below). |

Instructions add up from the root down. `delivery`, `max_output_tokens` and `think` come from the
nearest node on the path that sets them.

### Guards: `[when]`

Every rule that is set must pass; a node without rules always applies.

| Rule | Passes when |
|---|---|
| `app = ["slack.exe", "*teams*"]` | the process name matches a glob, ignoring case |
| `window_title = "(?i)inbox"` | the regular expression is found in the window title |
| `url = ["https://mail.google.com/*"]` | the browser address matches a glob |
| `role = ["Edit", "Document"]` | the focused element's role is one of these, ignoring case |
| `element_name = "(?i)reply"` | the regular expression is found in the focused element's name |
| `selection = true` | text is selected (`false`: nothing is) |
| `text = true` | the focused field holds text |
| `editable = true` | the focused element accepts typing |
| `transcript = "(?i)^translate"` | the regular expression is found in what the user said |

The inspector's Context tab shows every value these rules compare, for the window in front.

### Preferences: `[prefer]`

A guard decides whether a branch can be chosen at all; `[prefer]` decides when it should be,
without the model. It takes the same rules as `[when]`. When one candidate's `[prefer]` passes,
the parent's decision takes it with no model call, and the trace says which rule did
(`preferred: its transcript rule passed`). When several pass, the decision chooses among those
alone, as it would among all. A branch whose `[prefer]` fails stays a candidate, so a keyword
that is not said never takes a branch away.

```toml
# ask/decide.toml: "Pregunta: ¿qué dice Paul?" is always a question for jevons.
[prefer]
transcript = "(?i)^\\W*(pregunta|question)\\b"
```

Use it for what the user says on purpose (a keyword) and for contexts where the answer never
changes (a terminal is always a place to dictate). Leave softer signals, such as a field that does
not accept typing, to the model: the decision's state says whether the focused element accepts
typing, and the descriptions can say what that suggests.

## `decide.toml`

| Field | Meaning |
|---|---|
| `question` | What the decision model answers; the branches' descriptions are the choices. |
| `select` | `model` (default): the model chooses. `rules`: the highest `priority`, then the most specific guard; the model only breaks exact ties. |
| `fallback` | The branch taken when no guard passes, or the model is below `min_probability` or unavailable. It is taken even if its own guard fails. |
| `min_probability` | Below this probability (0 to 1) for the branch the model chose, run `enrich`, ask again, then take the fallback. The built-in root uses 0.7. |
| `enrich` | Extract and investigation names to read only when the first answer is unsure. |
| `branches` | Take the branches from a shared folder such as `"_actions"` instead of subfolders. |
| `only` | With `branches`: keep only these of them. |
| `steps`, `samples` | System One refinement steps (1 to 8) and samples (1 to 32). |

With one candidate the model is not asked. The model sees the context (the application, the
window, the focused element and whether it accepts typing, the text around the cursor and the
selection), what the user said and the investigations so far. Write descriptions for it: say who
the words are for, give the cues that tell branches apart, and add a short example or two. When a decision leads straight into another model decision, jevons asks
both in one request.

## `generate.toml` and `transcript.toml`

| Field | Meaning |
|---|---|
| `output` | `target` (default): into the application the take started in. `bubble`: shown by the tray icon, with Copy and Insert. `clipboard`. `none`. |
| `action` | With `target`: `insert` at the cursor (default), `replace` the selection, or `rewrite` the selection or whole field following what the user said. A transcript cannot rewrite. |
| `prompt` | `generate.toml` only: the model's input. By default: the context, the investigations and what the user said. |

Text goes into the target only if the same window is still in front once the user lets go of the
keys; otherwise it waits on the clipboard.

## `tool.toml`

Tools are registered in the desktop settings, never here: this folder can use a tool but not add
one. `TOOLS.md` lists the registered tools with their arguments.

```toml
description = "Save what the user says as a note"
tool = "notes:create_note"     # a built-in tool's name, or server:tool for an MCP server
output = "bubble"              # bubble (default), target, clipboard, none, or next

[args.title]
generate = "A title for the note of at most 8 words"

[args.folder]                  # labels, chosen by the decision model
choose = { inbox = "Unsorted", work = "About work", personal = "Anything else" }

[args.body]
value = "{transcript}"         # literal text with placeholders

[args.urgent]
noul = "Did the user say it is urgent?"
```

Each argument sets exactly one of `generate`, `choose`, `noul` and `value`. `type = "integer"`
(or `number`, `boolean`) reads `generate` and `value` text as that type. Calls ask the user in the
bubble before running unless the settings say that tool may run unconfirmed; `confirm = true`
asks even then. With `output = "next"`, the node has exactly one branch folder, which receives the
tool's result as `{result}`.

## `agent.toml`

```toml
description = "Questions that need looking something up before answering"
tools = ["search:web_search", "fs:*"]   # server:* allows every tool of that server
max_steps = 4                           # model turns before it must answer (1 to 16)
output = "bubble"
```

The agent can always call `investigate` (below). Tool calls ask for confirmation as in
`tool.toml`. `prompt` sets the task; by default it is the context and what the user said.

## Extracts

An extract reads part of the application's interface with an XPath expression. It is exact and
fast (tens of milliseconds) and costs no model call, so prefer it to an investigation whenever the
same elements hold the answer every time:

```toml
[extract.channels]
xpath = "//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]]/@name"
as = "list"               # text (default), list, count, exists or table
limit = 50                # the most matches kept (1 to 500)

[extract.messages]
xpath = "(//ListItem[starts-with(@automation_id, 'message-list_')][.//Text])[position() > last() - 10]"
as = "table"
fields = { author = ".//Button[1]/@name", text = "string(.//Text[last()])" }
lazy = true               # read only when a node at or below uses {messages} or enrich = ["messages"]
```

- **Elements** are named by role: the control types UI Automation reports (`Window`, `Pane`,
  `Group`, `Document`, `List`, `ListItem`, `Tree`, `TreeItem`, `Edit`, `Button`, `Text`,
  `Hyperlink`, `Image`, `ToolBar`, `Tab`, `TabItem`, `Menu`, `MenuItem`, `CheckBox`, `ComboBox`…).
  A misspelled role is an error that names the one meant.
- **Attributes:**
  - `@name`, `@value` and `@role`;
  - `@class`, which in browsers and Electron apps is the HTML class list (so `has-class(@class, 'ql-editor')` matches one class);
  - `@automation_id`;
  - `@enabled`, `@offscreen`, `@selected`, `@toggled`, `@expanded` (`'true'` or `'false'`);
  - on windows, `@app`, `@title` and `@front`.

  Names are in the user's language. Prefer classes and automation ids, which are the same in
  every language.
- **Paths:** `//` searches below, `/` goes to children, `..` to the parent. The axes are `ancestor::`,
  `following-sibling::`, `preceding-sibling::` and the other XPath 1.0 axes.
- **Positions:** `[1]` and `[last()]` count among siblings. `(//ListItem)[last()]` counts over the
  whole result, and is much faster than `//ListItem[last()]`.
- **Functions:** XPath 1.0's functions, plus `ends-with`, `lower-case`, `upper-case`,
  `matches(text, regex)` and `has-class`.
- **Values:** `string(.)` or an element in `as = "text"` gives an element's text together with its descendants'.
  Password fields are never read.
- **Where it starts:** the expression starts at the window the take started in. `/Window[@app='slack.exe']//…` reads
  another application's window: list it in `scope = ["slack.exe"]`. That also needs the
  user's permission in the settings.
- **What `//` covers:** an expression that starts with `//` searches every window the extract may read (the take's own and each `scope` window). One that starts with `.//` stays in the take's window.
- **Only in some applications:** `app = ["slack.exe"]` (process-name globs, any case) reads the
  extract only in takes from those applications; elsewhere nothing is read and its answer is
  empty. That lets the root declare an application's extracts once for every branch below: the
  built-in root reads Slack's `{slack_conversation}`, `{slack_messages}` and `{slack_channels}`.
- **Variables:** `$name` takes a placeholder's value, such as `//TreeItem[@name = $transcript]` or
  `$chat.name`. Values are never pasted into the expression, so they cannot change what it means.
- **Answers:** available at this node and below as `{channels}` and `{messages}` (JSON), and
  `{messages.author}` in templates. Decisions and generation prompts see them too.
- **Lists only hold what is on screen.** A list such as a chat's messages holds only the rows
  currently rendered.

To find and improve expressions, use the Context tab. **Read by the flow tree** reads every
extract that applies to the window in front (lazy ones included) and shows each answer. The
**Extracts** card takes any extract of this tree, or a new one: edit its expression, its `as` and
its table columns, and see its answer and what it matched, read as a take would (its `app`,
`scope` and `$variables` included). **Live** tries each edit after a pause in typing and again when
the window changes. **Save** writes the expression back into its node file (keeping the file's
comments) once the tree still loads, and **Copy as TOML** gives a new one to paste here. From a
shell: `jevons-desktop --xpath "//ListItem" --app slack.exe`. An investigation that succeeds
shows the XPath it remembered in the take's trace, and that XPath can become an extract.

## `run.toml`

Automations are tasks the user recorded once, in the automations library next to the settings
(see its `AGENTS.md`). A run node runs one of them:

```toml
description = "The user asks to run one of their saved automations"
automations = ["slack-post", "jira-ticket"]   # which it may run; empty or ["*"] for all
output = "bubble"                              # bubble (default), target, clipboard, none, or next
```

- **Which one:** with several approved automations, the decision model picks one by their
  descriptions.
- **Arguments:** yes/no arguments are answered by the decision model. The others are written by
  the language model from what the user said, following each argument's description.
- **Before it runs:** it asks in the bubble first, unless the settings list it as unconfirmed.
  Only automations the user approved run. The built-in tree's `run` branch runs any of them.

## Investigations

The context investigator reads the application's interface (UI Automation on Windows) to answer a
question in a fixed shape:

```toml
[investigate.chat]
question = "Which conversation is open, and what are its last messages?"
schema = { name = "string", messages = [{ author = "string", text = "string" }] }
scope = ["slack.exe"]     # applications it may read; by default only the one in front
max_steps = 8             # navigation steps (1 to 32)
lazy = true               # run only when a node at or below uses {chat...} or enrich = ["chat"]
```

Schema types: `"string"`, `"number"`, `"integer"`, `"boolean"`, `"a | b | c"` (one of these
labels), `["shape"]` (a list of one shape) and tables (objects). Any field may come back `null`.
The answer is available at this node and below as `{chat}` (JSON) and `{chat.name}`. Answers are
reused within a take, so declaring the same question on two nodes costs one investigation.
Reading windows other than the one in front also needs the user's permission in the settings.

## Placeholders

`prompt`, `question`, inline `instructions`, tool arguments and investigation questions may use:

| Placeholder | Value |
|---|---|
| `{transcript}` | what the user said |
| `{selection}`, `{field_text}`, `{before_caret}`, `{after_caret}` | the focused field's text |
| `{app}`, `{window}`, `{url}`, `{field}` | where the user is |
| `{clipboard}` | the clipboard text, when the settings allow reading it |
| `{context}` | all of the above, described for a model |
| `{route}` | the branches taken so far, such as `dictate/chat` |
| `{name}`, `{name.field}` | an extract or investigation declared here or above |
| `{result}`, `{result.field}` | below a tool or agent with `output = "next"` |

Write `{{` and `}}` for literal braces. `instructions.md` is plain prose: no placeholders.

## Rules the loader enforces

- One node file per folder; unknown fields are errors, reported with their line.
- A decision has 1 to 128 branches, each with a `description`. `fallback` names one of them.
  If every branch has a guard, a `fallback` is required.
- Leaves (`generate.toml`, `transcript.toml`) have no branch folders. Tools, agents and runs have
  one only with `output = "next"`.
- A run node's `automations` name automations in the library.
- `branches` names a `_` folder under this root; shared folders cannot lead back to themselves.
- Every placeholder and `$variable` resolves where it is used; `enrich` names extracts and
  investigations in scope. Every XPath expression parses; an error gives its column.
- Folders nest at most 8 deep, and a path takes at most 4 model decisions: each costs a model
  call while the user waits.
- Tool and agent nodes name registered tools, and tool arguments match the tool's schema.

## Example: a branch for translating

Put it at the root, next to `dictate` and `ask`, where the model chooses by what the user wants
(`dictate` chooses by application, with rules):

```toml
# translate/generate.toml
description = "The user asks to translate the selected text into another language"
action = "rewrite"
instructions = "Translate the text into the language the user names. Output only the translation."

[when]
selection = true
transcript = "(?i)(translate|traduc)"
```

The guard keeps it out of the root decision unless text is selected and the user said
"translate", so ordinary dictation never pays for it.
