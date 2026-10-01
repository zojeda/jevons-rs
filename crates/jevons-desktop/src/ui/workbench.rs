//! The extract workbench on the Context tab: pick any `[extract]` of the flow tree (or start a
//! new one), edit its expression, answer type and table columns, and see what it reads in the
//! window the tab shows, as a take would read it. Live mode tries each edit after a pause in
//! typing and again when the window changes; Save writes the edit into its node file.

use super::Ctx;
use super::components::{Choice, Collapsible, Select, Switch, copy};
use crate::agent::{Command, TrialRequest};
use dioxus::prelude::*;
use jevons_desktop_core::flow::extract;
use jevons_desktop_core::flow::spec::{ExtractAs, ExtractSpec};
use std::collections::BTreeMap;

const KINDS: [(ExtractAs, &str); 5] = [
    (ExtractAs::Text, "text"),
    (ExtractAs::List, "list"),
    (ExtractAs::Count, "count"),
    (ExtractAs::Exists, "exists"),
    (ExtractAs::Table, "table"),
];

/// An extract of the tree, as the selector names it: its node's path and its name.
pub fn key(path: &str, name: &str) -> String {
    format!("{path}\u{1f}{name}")
}

/// Table columns as the editor shows them: `column = expression`, one per line.
fn fields_text(fields: &BTreeMap<String, String>) -> String {
    fields
        .iter()
        .map(|(column, expression)| format!("{column} = {expression}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_fields(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(column, expression)| (column.trim().to_string(), expression.trim().to_string()))
        .filter(|(column, expression)| !column.is_empty() && !expression.is_empty())
        .collect()
}

/// `chosen` selects the extract to edit (a [`key`], or `None` for a new expression); the
/// "Read by the flow tree" card sets it too. `draft` brings an expression from the interface
/// browser: it becomes a new expression, tried at once. `rev` changes with every repaint, so a
/// trial's answer shows as it arrives.
#[component]
pub fn Workbench(
    rev: u64,
    chosen: Signal<Option<String>>,
    draft: Option<Signal<Option<String>>>,
) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let view = ctx.view.lock().expect("the view lock");
    let flows = view.flows.clone();
    let trial = view.trial.clone();
    let trying = view.trying;
    let saved = view.saved.clone();
    drop(view);

    let mut name = use_signal(String::new);
    let mut xpath = use_signal(String::new);
    let mut kind = use_signal(|| ExtractAs::Text);
    let mut fields = use_signal(String::new);
    // The chosen extract's other settings (scope, app, limit, lazy), kept as they are.
    let mut base = use_signal(|| None::<ExtractSpec>);
    let mut file = use_signal(|| None::<String>);
    let mut live = use_signal(|| false);

    // Every extract of the tree, by node, for the selector.
    let mut choices = vec![Choice {
        value: None,
        label: "New expression".into(),
    }];
    for node in flows.nodes() {
        for extract_name in node.extracts.keys() {
            let path = if node.path.is_empty() {
                "/"
            } else {
                &node.path
            };
            choices.push(Choice {
                value: Some(key(&node.path, extract_name)),
                label: format!("{path} · {extract_name}"),
            });
        }
    }

    // Loads the chosen extract into the editor when the choice changes, from the tree as it is
    // then (a save reloads it).
    let tree = ctx.clone();
    use_effect(move || {
        let picked = chosen();
        let flows = tree.view.lock().expect("the view lock").flows.clone();
        let entry = picked.as_ref().and_then(|k| {
            flows.nodes().iter().find_map(|node| {
                node.extracts
                    .iter()
                    .find(|(extract_name, _)| key(&node.path, extract_name) == *k)
                    .map(|(extract_name, e)| {
                        (node.file.clone(), extract_name.clone(), e.spec.clone())
                    })
            })
        });
        match entry {
            Some((node_file, extract_name, spec)) => {
                name.set(extract_name);
                xpath.set(spec.xpath.clone());
                kind.set(spec.kind);
                fields.set(fields_text(&spec.fields));
                file.set(Some(node_file));
                base.set(Some(spec));
            }
            None => {
                file.set(None);
                base.set(None);
            }
        }
    });

    // An expression from the interface browser: a new expression, as text, tried at once.
    let drafts = ctx.clone();
    use_effect(move || {
        let Some(mut draft) = draft else {
            return;
        };
        let Some(expression) = draft() else {
            return;
        };
        draft.set(None);
        chosen.set(None);
        name.set(String::new());
        xpath.set(expression.clone());
        kind.set(ExtractAs::Text);
        fields.set(String::new());
        drafts.send(Command::TryExtract(Box::new(TrialRequest {
            file: None,
            name: String::new(),
            spec: ExtractSpec {
                xpath: expression,
                kind: ExtractAs::Text,
                fields: BTreeMap::new(),
                limit: None,
                scope: Vec::new(),
                app: Vec::new(),
                lazy: false,
            },
            debounce: false,
        })));
    });

    let request = move |debounce: bool| {
        let mut spec = base().unwrap_or_else(|| ExtractSpec {
            xpath: String::new(),
            kind: ExtractAs::Text,
            fields: BTreeMap::new(),
            limit: None,
            scope: Vec::new(),
            app: Vec::new(),
            lazy: false,
        });
        spec.xpath = xpath();
        spec.kind = kind();
        spec.fields = if kind() == ExtractAs::Table {
            parse_fields(&fields())
        } else {
            BTreeMap::new()
        };
        Box::new(TrialRequest {
            file: file(),
            name: name(),
            spec,
            debounce,
        })
    };
    let (edits, toggle, run, save, kinds, columns) = (
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
    );
    let settings = base().map(|spec| {
        let mut parts = Vec::new();
        if !spec.app.is_empty() {
            parts.push(format!("only in {}", spec.app.join(", ")));
        }
        if !spec.scope.is_empty() {
            parts.push(format!("also reads {}", spec.scope.join(", ")));
        }
        if let Some(limit) = spec.limit {
            parts.push(format!("at most {limit}"));
        }
        if spec.lazy {
            parts.push("lazy".into());
        }
        parts.join(" · ")
    });
    let current = xpath();
    let stale = trial.as_ref().is_some_and(|t| t.xpath != current);
    let can_save = file().is_some() && !xpath().trim().is_empty();
    let save_label = file().map_or_else(String::new, |f| format!("Save to {f}"));

    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Extracts" }
                    div { class: "dx-card-description",
                        "Try any [extract] of the flow tree, or a new one, on this window as a take reads it; \
                         edit it and save it back"
                    }
                }
                Switch {
                    checked: live(),
                    label: "Live".to_string(),
                    onchange: move |on: bool| {
                        live.set(on);
                        toggle.send(Command::LiveTrials(on));
                        if on && !xpath().trim().is_empty() {
                            toggle.send(Command::TryExtract(request(false)));
                        }
                    },
                }
            }
            div { class: "dx-card-content",
                div { class: "row",
                    Select {
                        value: chosen(),
                        choices,
                        onchange: move |picked: Option<String>| chosen.set(picked),
                    }
                    if file().is_none() {
                        input { class: "dx-input mono", placeholder: "name",
                            value: "{name}",
                            oninput: move |e| name.set(e.value())
                        }
                    }
                }
                if let Some(settings) = settings.filter(|s| !s.is_empty()) {
                    p { class: "muted", "{settings}" }
                }
                textarea { class: "dx-input mono workbench-xpath", rows: "3",
                    placeholder: "//ListItem[last()]",
                    value: "{xpath}",
                    oninput: move |e| {
                        xpath.set(e.value());
                        if live() {
                            edits.send(Command::TryExtract(request(true)));
                        }
                    }
                }
                div { class: "row",
                    div { class: "dx-tabs-list",
                        {KINDS.iter().map(|(k, label)| {
                            let (k, kinds) = (*k, kinds.clone());
                            rsx! {
                                button { key: "{label}", class: "dx-tabs-trigger",
                                    "data-state": if kind() == k { "active" } else { "inactive" },
                                    onclick: move |_| {
                                        kind.set(k);
                                        if live() {
                                            kinds.send(Command::TryExtract(request(false)));
                                        }
                                    },
                                    "{label}"
                                }
                            }
                        })}
                    }
                    button { class: "dx-button", "data-size": "sm",
                        disabled: xpath().trim().is_empty(),
                        onclick: move |_| run.send(Command::TryExtract(request(false))),
                        "Try"
                    }
                    if can_save {
                        button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                            onclick: move |_| save.send(Command::SaveExtract(request(false))),
                            "{save_label}"
                        }
                    }
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        disabled: xpath().trim().is_empty(),
                        title: "Copy [extract.<name>] as TOML, to paste into a node file",
                        onclick: move |_| {
                            let spec = request(false).spec;
                            let label = if name().trim().is_empty() { "new".to_string() } else { name().trim().to_string() };
                            copy(&extract::as_toml(&label, &spec));
                        },
                        "Copy as TOML"
                    }
                }
                if kind() == ExtractAs::Table {
                    textarea { class: "dx-input mono workbench-fields", rows: "3",
                        placeholder: "author = .//Button[1]/@name\ntext = string(.//Text[last()])",
                        value: "{fields}",
                        oninput: move |e| {
                            fields.set(e.value());
                            if live() {
                                columns.send(Command::TryExtract(request(true)));
                            }
                        }
                    }
                }
                match saved {
                    Some(Ok(done)) => rsx! { p { class: "muted", "{done}" } },
                    Some(Err(problems)) => rsx! {
                        div { class: "stack",
                            p { class: "error-text", "Not saved: the tree would not load" }
                            {problems.iter().map(|p| rsx! { p { class: "error-text mono", "{p}" } })}
                        }
                    },
                    None => rsx! {},
                }
                {results(trial, trying, stale)}
            }
        }
    }
}

/// What the last trial found: its problems, or its answer and what it matched.
fn results(trial: Option<crate::agent::TrialView>, trying: bool, stale: bool) -> Element {
    let Some(t) = trial else {
        return rsx! {
            if trying {
                p { class: "muted", "Reading…" }
            }
        };
    };
    let summary = match &t.trial.found {
        Some(found) => {
            let mut parts = vec![
                match found.matches {
                    1 => "1 match".to_string(),
                    n => format!("{n} matches"),
                },
                format!("{} ms", t.trial.ms),
                format!("in {}", t.window),
            ];
            if !t.trial.read_first.is_empty() {
                parts.push(format!("read first: {}", t.trial.read_first.join(", ")));
            }
            parts.join(" · ")
        }
        None => String::new(),
    };
    let answer = t.trial.found.as_ref().map(|f| shown(&f.value));
    let note = t.trial.found.as_ref().and_then(|f| f.note.clone());
    let matched = t.trial.matched.join("\n");
    rsx! {
        if trying {
            p { class: "muted", "Reading…" }
        } else if stale {
            p { class: "muted", "(for an earlier version of the expression)" }
        }
        {t.trial.errors.iter().map(|e| rsx! { p { class: "error-text mono", "{e}" } })}
        if !summary.is_empty() {
            p { class: "muted", "{summary}" }
        }
        if let Some(note) = note {
            p { class: "warn", "{note}" }
        }
        if let Some(answer) = answer {
            if answer.is_empty() {
                p { class: "muted", "(nothing)" }
            } else {
                pre { class: "code", "{answer}" }
            }
        }
        if !matched.is_empty() {
            Collapsible { title: "What it matched".to_string(), subtitle: None, open: false,
                pre { class: "code", "{matched}" }
            }
        }
    }
}

/// An answer as text: a string as it is, a list of strings one per line, anything else as JSON.
fn shown(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) if items.iter().all(Value::is_string) => items
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("\n"),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_columns_round_trip_through_the_editor() {
        let fields: BTreeMap<String, String> = [
            ("author".to_string(), ".//Button[1]/@name".to_string()),
            ("text".to_string(), "string(.//Text[last()])".to_string()),
        ]
        .into();
        let text = fields_text(&fields);
        assert_eq!(parse_fields(&text), fields);
        // `=` inside an expression stays in it; lines without one are ignored.
        let parsed = parse_fields("n = count(.//*[@a='b'])\n\nnot a column\n");
        assert_eq!(parsed["n"], "count(.//*[@a='b'])");
        assert_eq!(parsed.len(), 1);
    }
}
