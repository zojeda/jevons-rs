# Script API

An automation's `script.rhai` is [Rhai](https://rhai.rs/book/). It reads the interface of its
applications (the manifest's `apps`) with XPath and acts on what it finds. It has no file,
network, process or module access. Its last expression is its answer, fitted to the manifest's
`returns`:

```rhai
step("opening the channel");
find("//TreeItem[.//Text[@name = $channel]]").invoke();
let composer = wait_for("//Edit[has-class(@class, 'ql-editor')]", 3000);
composer.type_text(args.message);
press("enter");
#{ posted: true }
```

## Arguments

`args` holds the call's arguments, checked and typed by the manifest: `args.channel`. Inside
an XPath expression, `$channel` is the same argument.

## Reading the interface

Each query evaluates an XPath expression in the front window of the automation's applications.
The same queries are methods of a `Window` (from `window()`) and of an `Element` (relative to it,
such as `row.find(".//Button[1]")`). Every query takes an optional map of `$variables`:
`find("//TreeItem[@name = $c]", #{ c: name })`.

| Function | Returns |
|---|---|
| `find(xpath)` | The one `Element` that matches. It fails with `not_found` when none does, and with `ambiguous` when several do. |
| `find_all(xpath)` | An array: the matching elements, or the texts of matching attributes. |
| `try_find(xpath)` | The one matching `Element`, or `()` when none matches. It fails with `ambiguous`. |
| `exists(xpath)` | Whether anything matches (or the value of a yes/no expression). |
| `text(xpath)` | The expression's text, such as `text("//Tree/@name")` or `text("count(//ListItem)")`. |
| `wait_for(xpath, ms)` | Waits up to `ms` milliseconds for a match, then returns it as `find` does. It fails with `timeout`. |
| `wait_gone(xpath, ms)` | Waits up to `ms` milliseconds until nothing matches. It fails with `timeout`. |
| `window(app)` | The `Window` of an application of the automation, by process-name glob, such as `window("slack.exe")`. |
| `windows()` | The open windows of the automation's applications, the front one first. |

XPath in short:
- **Elements** are named by role (`List`, `ListItem`, `Tree`, `TreeItem`, `Edit`, `Button`, `Text`, `Group`, `Document`…).
- **Attributes:** `@name`, `@value`, `@class`, `@automation_id`, `@role`, `@enabled`, `@selected`, `@toggled`, `@expanded`, `@offscreen`.
- **Paths:** `//` searches below, `/` goes to children, `..` to the parent.
- **Functions:** `has-class(@class, 'x')` matches one class of a list, and `contains`, `starts-with` and `normalize-space` work as in XPath 1.0.
- **Positions:** `(//ListItem)[last()]` is the last match over the whole result.

Names are in the user's language, so prefer classes and automation ids. The flows folder's
`AGENTS.md` has the full vocabulary.

## Elements

| Property | |
|---|---|
| `name`, `role`, `value`, `class`, `automation_id` | As the accessibility layer reports them; empty when absent. A password field's `value` is always empty. |
| `enabled` | Whether it takes input. |
| `app` | The application it is in. |
| `text` | Its text with its descendants', as XPath's `string()` reads it. |

| Action | |
|---|---|
| `invoke()` | Its own action (a button's press, a link's follow, a list item's selection); a click when it has none. |
| `click()` | A mouse click at its clickable point. |
| `focus()` | Moves the keyboard focus to it. |
| `set_value(text)` | Replaces its value through the accessibility layer. |
| `type_text(text)` | Focuses it and types the text. |
| `toggle()`, `select()`, `expand()`, `collapse()` | Through its control pattern. |
| `scroll_into_view()` | Scrolls its container until it shows. |

Actions fail with `denied` outside the automation's applications and for text into password
fields, and with `action_failed` when the element cannot do it.

## Windows

| | |
|---|---|
| `app`, `title` | The window's application and title. |
| `activate()` | Brings it to the front. |

## Keys and typing

| Function | |
|---|---|
| `press(chord)` | Presses keys together in the window in front, such as `press("ctrl+k")`, `press("enter")` or `press("shift+tab")`. |
| `type_text(text)` | Types into the window in front. |

They fail with `denied` when the window in front is not one of the automation's applications,
so keys never land in another application.

## Control

| Function | |
|---|---|
| `step(label)` | Tells the user what the automation does now, in the bubble and the trace. |
| `log(value)` | Adds a line to the run's trace (never the log file). `print` and `debug` do the same. |
| `sleep(ms)` | Waits, at most 10 s at a time. Prefer `wait_for`, which stops waiting as soon as it can. |
| `confirm(message)` | Asks the user yes or no in the bubble, and returns their answer. |
| `fail(message)`, `fail(kind, message)` | Stops with an error, of kind `script` or the one given. |

## Errors

A failure is a map: `#{ kind, message }`, plus `xpath` and `count` for queries. Its kind is one of:

| Kind | Meaning |
|---|---|
| `not_found` | Nothing matched. |
| `ambiguous` | Several elements matched where `find` needs one. |
| `timeout` | A wait ran out, or the whole run did. |
| `denied` | The action was refused. |
| `action_failed` | The element could not do the action. |
| `cancelled` | The user cancelled the run. |
| `invalid` | A mistake: not an XPath expression, or a `$variable` with no value. |
| `platform` | The interface could not be read. |

`try`/`catch` handles them:

```rhai
let row = ();
try {
    row = find("//ListItem[@automation_id = $id]");
} catch (e) {
    if e.kind != "not_found" { throw e; }
    fail("not_found", `no message ${args.id}`);
}
```

## Rhai in short

- **Values:** maps are `#{ a: 1 }` (not `{ a: 1 }`), arrays are `[1, 2]`, and text with values is `` `hello ${name}` ``.
- **Variables:** `let x = 1;` declares one. Using an undeclared one is an error, and so is declaring the same name twice in one scope.
- **Control flow:** `if`, `for x in array`, `while` and `loop` with `break`.
- **Functions:** `fn name(a) { ... }` declares one. A function sees only its parameters: pass it what it needs.
- **Not available:** `eval`, `import`, `Fn` and `call`.
