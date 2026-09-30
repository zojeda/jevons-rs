# Automations: research and measurements

This note backs the design of jevons-desktop's programmable layer:
- XPath queries over accessibility trees (`[extract]` in flow nodes);
- actions on the elements they find;
- Rhai automation scripts that an agent writes from a recorded demonstration.

It records what was measured and why each choice was made. [desktop.md](desktop.md) documents the features themselves.

## Measurements (Windows, 2026-09-30)

These were taken with a read-only PowerShell probe over .NET UI Automation, against Slack 4.52.162 (Electron, window class `Chrome_WidgetWin_1`) showing a direct-message conversation. The control view held 383 elements.

| How | Elements | Time |
|---|---:|---:|
| Control-view `TreeWalker`, four live properties per element (what the UIA worker did until now) | 383 | 263 ms |
| `FindAll(Descendants, True)` with a cache request (4 properties, bounds, offscreen) | 382 | 70–79 ms |
| `GetUpdatedCache` with `TreeScope.Subtree` (one call), then walking `CachedChildren` in process | 383 | 17 ms fetch, 20 ms total |
| `FindAll(Descendants, ControlType = TreeItem)`, cached | 28 | 16 ms |
| `FindAll(Descendants, ControlType = ListItem)`, cached | 33 | 17 ms |

A cached subtree fetch is about 15 times faster than the live walk, and a native condition returns in one call. The XPath planner therefore turns steps into `FindAll` calls with conditions, and snapshots use one subtree cache request. The live walk costs about 0.7 ms per element. That is the same order as the 1.1 s measured earlier for a ~1300-element Slack view, and FlaUI's XPath has the same per-move design.

The same window, through jevons' own evaluator (`jevons-desktop --xpath … --app slack.exe`, release build, times of the evaluation alone):

| Expression | Result | Time |
|---|---:|---:|
| `count(//*)` | 382 | 106 ms |
| `count(//TreeItem)` | 28 | 27 ms |
| `count(//ListItem[starts-with(@automation_id, 'message-list_')])` | 28 | 32 ms |
| `//Edit[has-class(@class,'ql-editor')]/@class` | 1 | 22 ms |
| the selected conversation's name (built-in `ask/slack`) | 1 | 21 ms |
| the last 15 message rows' names (built-in `ask/slack`) | 15 | 44 ms |
| `//TreeItem[1]/@class` (positional: one children read per node) | 3 | 310 ms |

A search with native conditions takes 20–50 ms. A positional step after `//`, such as `//TreeItem[1]`, reads the children of every element and costs ten times as much; `(//TreeItem)[1]` does not.

## Slack's interface, as UI Automation reports it

- **Names are localized.** With Slack in Spanish, the channel tree is named `Canales y mensajes directos`, the composer `Mensaje a <name>` and the toolbar `Navegación histórica`. Selectors that rely on names break when the language changes.
- **Chromium reports the HTML `class` attribute as ClassName**, with every class separated by spaces (`ql-editor ql-blank`, `c-link c-timestamp`). These classes are the same in every language, so they make the most stable selectors. The XPath dialect has `has-class(@class, 'ql-editor')` for matching a single class.
- **Messages** are `ListItem`s whose automation id is `message-list_<timestamp>`. There are also `message-list_unreadDivider`, `message-list_bottomSpacer` and date dividers. Each message holds a `Document` with class `c-message_kit__hover`, and inside that a timestamp `Hyperlink` (`c-link c-timestamp`), the text and the reactions. Messages support Invoke and ScrollItem. The list is virtualized (`c-virtual_list__item`), so only rendered rows exist.
- **Messages are grouped.** A message that follows one from the same author has no sender button: of 24 rendered rows, 12 had one. Every row has its time, in a `Hyperlink` with class `c-link c-timestamp`. A row's name summarizes it (author, text, time, reactions) in a format that varies with the grouping, so `ask/slack` reads the names, not separate columns.
- **Channels** are `TreeItem`s under a `Tree` with class `c-virtual_list__scroll_container`. Every item has an automation id:
  - Slack's own ids for channels and direct messages;
  - `unified_directory`, `Vall_threads`, `Pdrafts` and `Pbrowse-huddles` for the fixed entries;
  - `sectionHeading-<id>` for section headings.

  A channel item holds a `Group` with class `p-channel_sidebar__channel`. The open conversation's item has the class `p-channel_sidebar__static_list__item--selected`. UI Automation's `SelectionItem.IsSelected` stays false, and the item's own name can be empty, so the conversation's name is the item's text. The items support SelectionItem, ExpandCollapse and ScrollItem.
- **The composer** is an `Edit` with class `ql-editor` (Quill) that supports Value, Text and ScrollItem. The channel filter is an `Edit` with class `c-filter_input__input`.
- **The web content** starts at a `Document` with automation id `RootWebArea`, below seven native `Pane`s (`RootView`, `NonClientView`, `WinFrameView`, `ClientView`, `View`…).

`examples/desktop/trees/slack.json` is a synthetic tree with this structure. Every name and message in it is made up.

## XPath engine

Rust has no XPath engine that both evaluates lazily over a live tree and exposes its steps for planning:

| Crate | Tree | Why not |
|---|---|---|
| `servo-xpath` 0.6 (MPL-2.0) | caller's traits | Its AST is closed (`pub(crate)` steps), so we can't plan `FindAll` from it, and a leading `//` walks everything |
| `xee-xpath` 0.1 (XPath 3.1) | `xot` documents | It needs its own materialized DOM |
| `sxd-xpath` 0.4, `amxml` 0.5 | their own DOM | Abandoned since 2018 |
| `xrust` 2.2 | a 57-method `Node` trait | It needs mutation methods a read-only live tree cannot provide |

jevons has its own XPath 1.0 subset in `jevons-desktop-core::xpath`. Its design follows the NovaWindows Appium driver, which turns child and descendant steps into UIA `FindAll` with ControlType and property conditions and filters positions and functions on the client. Element names are the language-independent ControlType names (`ListItem`), never `LocalizedControlType`. From Playwright's locators it borrows role + name as the primary selector and strict single-match lookups: a script's `find` fails when zero elements match, and when two or more do.

## UI Automation from Rust

The `uiautomation` 0.25.1 crate exposes everything needed without `unsafe` in our code:
- **patterns:** Invoke, Value, Toggle, SelectionItem, ExpandCollapse, ScrollItem, LegacyIAccessible `do_default_action`;
- **element calls:** `set_focus`, bounding rectangles and clickable points;
- **search:** `find_all` with property conditions;
- **caching:** cache requests (`find_all_build_cache`, `build_updated_cache`, `get_cached_*`);
- **recording:** `element_from_point`, and event handlers for focus, Invoke, selection, property changes and window opening.

Events need the crate's `event` feature. Their handlers run on UI Automation's own threads and only require `'static`, so they must capture only `Send` values and send plain data out.

The crate's `UIMatcher` walks the tree itself, one live call per property, so XPath does not use it.

## Scripting language

The candidates were Rhai, Rune, Starlark, Koto and Boa. We chose **Rhai 1.26**:
- **Build and threads:** it is pure Rust, and its `sync` feature makes it `Send + Sync`.
- **Limits:** it caps operations, call depth and sizes, and `on_progress` enforces a deadline and cancellation.
- **Checks:** strict variables, and parse errors with a line and column. Function calls are checked by walking the AST.
- **Errors:** scripts can use `try`/`catch`/`throw` on typed error maps.
- **Host data:** it converts to and from `serde_json::Value`.
- **Documentation:** its signature metadata generates the API reference that scripts and the author agent read.

| Rejected | Reason |
|---|---|
| Rune 0.14 | Values are `!Send`; no try/catch; slow releases |
| Starlark 0.14 | No `try` or `while`; blake3 compiles C/asm |
| Koto 0.16 | Its standard library has file I/O and `os.command` |
| Boa 0.22 | No instruction limit or interrupt; `!Send` |
| mlua, rquickjs, deno_core | C or C++ builds |
| wasmtime / extism | Scripts would be a compiled language with a build step |

## How recording works

- **Clicks** come from the low-level hook the app already runs for held hotkeys. The UI Automation thread finds the element under each point (`ElementFromPoint`, normalized to the control view) and its window.
- **Keys** come from the same hook. Characters typed into the focused field become one step, and chords (with ctrl, alt or meta, and named keys such as Enter) are key presses.
- **The app's own input** is marked as its own for a moment and not reported: its delivery, its clicks and its keys.
- **The interface before each step** is one cached subtree read of the window, taken once the previous step has settled (400 ms).
- **UI Automation's own events** (focus changed, Invoke) are not used yet: the hook and the focused element are enough to record the steps.

## Still to measure

- A single `--xpath` run once found no Slack window, and the same command found it right after. Window listing skips windows with an empty title, and Slack's title may briefly be empty.
- Whether `element_from_point` matches the hook's coordinates on scaled displays (DPI awareness), and whether clicks through SendInput land where the element's clickable point is.
- Whether `set_focus` or `WindowPattern` brings a window forward when called from the tray app, since Windows' foreground lock may refuse. Scripts' `activate()` depends on it.
