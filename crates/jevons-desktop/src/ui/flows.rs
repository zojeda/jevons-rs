//! The flow tree: its nodes, the problems of the flows folder, and new branches drafted from the
//! current context.

use super::Ctx;
use super::components::{Collapsible, badge};
use crate::agent::{Command, open_folder};
use dioxus::prelude::*;
use jevons_desktop_core::flow::guard::draft_when;
use jevons_desktop_core::flow::spec::Select;
use jevons_desktop_core::flow::tree::NodeSpec;
use jevons_desktop_core::flow::{Kind, defaults};

/// A node as the page lists it.
#[derive(Clone, PartialEq)]
struct Row {
    depth: usize,
    path: String,
    title: String,
    kind: Kind,
    description: String,
    details: String,
    file: String,
    text: String,
}

#[component]
pub fn FlowsPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let mut parent = use_signal(|| "dictate".to_string());
    let mut name = use_signal(String::new);
    let mut message = use_signal(|| None::<String>);
    let view = ctx.view.lock().expect("the view lock");
    let dir = view.config.flows_dir(&view.config_file);
    let errors: Vec<String> = view.flow_errors.iter().map(ToString::to_string).collect();
    let notes = view.flow_notes.clone();
    let tool_problems = view.tool_problems.clone();
    let tree = view.flows.clone();
    let context = view.context.clone();
    drop(view);

    let builtin = tree.source == defaults::BUILTIN;
    let rows: Vec<Row> = tree
        .outline()
        .into_iter()
        .map(|(depth, id)| {
            let node = tree.node(id);
            let mut details = Vec::new();
            match &node.spec {
                NodeSpec::Decide(d) => {
                    details.push(match d.select {
                        Select::Model => "the model chooses".to_string(),
                        Select::Rules => "rules choose".to_string(),
                    });
                    if let Some(fallback) = &d.fallback {
                        details.push(format!("fallback {fallback}"));
                    }
                    if let Some(branches) = &d.branches {
                        details.push(format!("branches from {branches}"));
                    }
                }
                other => {
                    if let Some(output) = other.output() {
                        details.push(format!("→ {output:?}").to_lowercase());
                    }
                }
            }
            let rules = node.guard.specificity();
            if rules > 0 {
                details.push(format!(
                    "{rules} guard rule{}",
                    if rules == 1 { "" } else { "s" }
                ));
            }
            if node.spec.common().priority != 0 {
                details.push(format!("priority {}", node.spec.common().priority));
            }
            if !node.investigations.is_empty() {
                let names: Vec<&str> = node.investigations.keys().map(String::as_str).collect();
                details.push(format!("investigates {}", names.join(", ")));
            }
            let text = if builtin {
                defaults::TREE
                    .iter()
                    .find(|(path, _)| *path == node.file)
                    .map(|(_, text)| text.to_string())
                    .unwrap_or_default()
            } else {
                std::fs::read_to_string(dir.join(&node.file)).unwrap_or_default()
            };
            Row {
                depth,
                path: node.path.clone(),
                title: if node.name.is_empty() {
                    "/".into()
                } else {
                    node.name.clone()
                },
                kind: node.kind(),
                description: node.description().unwrap_or_default().to_string(),
                details: details.join(" · "),
                file: node.file.clone(),
                text,
            }
        })
        .collect();

    let parent_path = parent();
    let id = name();
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    let parent_ok = tree
        .find(&parent_path)
        .is_some_and(|p| tree.node(p).kind() == Kind::Decide);
    let shared = tree.nodes().iter().any(|n| n.path.starts_with("_actions/"));
    let draft = context.as_ref().map(|c| {
        let title = if c.window.title.is_empty() { c.app.process_name.clone() } else { c.window.title.clone() };
        if shared {
            format!(
                "description = {title:?}\npriority = 10\nbranches = \"_actions\"\ninstructions = \"Describe how text should read here.\"\n\n{}",
                draft_when(c)
            )
        } else {
            format!("description = {title:?}\noutput = \"target\"\n\n{}", draft_when(c))
        }
    });
    let file_name = if shared {
        "decide.toml"
    } else {
        "generate.toml"
    };
    let open_dir = dir.clone();
    let reload = ctx.clone();

    rsx! {
        div { class: "spread",
            div { class: "stack",
                h2 { "Flow tree" }
                span { class: "muted mono", "{dir.display()}" }
            }
            div { class: "row",
                button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                    onclick: move |_| {
                        let _ = std::fs::create_dir_all(&open_dir);
                        open_folder(&open_dir);
                    },
                    "Open folder"
                }
                button { class: "dx-button", "data-style": "secondary", "data-size": "sm",
                    onclick: move |_| reload.send(Command::ReloadFlows),
                    "Reload"
                }
            }
        }
        if !errors.is_empty() {
            p { class: "warn",
                if builtin { "The flows folder has problems: the built-in tree runs until they are fixed." }
                else { "The flows folder has problems: the last tree that loaded cleanly still runs." }
            }
            {errors.iter().map(|e| rsx! { p { class: "error-text mono", "{e}" } })}
        }
        {notes.iter().map(|n| rsx! { p { class: "muted", "{n}" } })}
        {tool_problems.iter().map(|n| rsx! { p { class: "warn", "{n}" } })}
        p { class: "muted",
            "Every folder is a node; AGENTS.md in the folder explains the format to people and agents, \
             and TOOLS.md lists the tools the settings register. Files reload as soon as they are saved."
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "New branch from the current context" }
                    div { class: "dx-card-description", "Its guard matches the application, page and field the Context tab shows" }
                }
            }
            div { class: "dx-card-content",
                match (context, draft) {
                    (Some(context), Some(draft)) => {
                        let create_ctx = ctx.clone();
                        let dir = dir.clone();
                        let text = draft.clone();
                        rsx! {
                            p { class: "muted", "Matches {context.app.process_name} · {context.window.title}" }
                            div { class: "row",
                                input { class: "dx-input", placeholder: "under, such as dictate", value: "{parent_path}",
                                    oninput: move |e| parent.set(e.value().trim().trim_matches('/').to_string()) }
                                input { class: "dx-input", placeholder: "branch name, such as slack", value: "{id}",
                                    oninput: move |e| name.set(e.value()) }
                                button { class: "dx-button", "data-style": "accent", "data-size": "sm", disabled: !valid || !parent_ok,
                                    onclick: move |_| {
                                        let folder = dir.join(parent()).join(name());
                                        let file = folder.join(file_name);
                                        message.set(Some(if folder.exists() {
                                            format!("{} already exists", folder.display())
                                        } else {
                                            match std::fs::create_dir_all(&folder).and_then(|()| std::fs::write(&file, &text)) {
                                                Ok(()) => {
                                                    create_ctx.send(Command::ReloadFlows);
                                                    open_folder(&folder);
                                                    format!("Created {}; describe it and adjust its guard", file.display())
                                                }
                                                Err(e) => e.to_string(),
                                            }
                                        }));
                                    },
                                    "Create"
                                }
                            }
                            if !parent_ok {
                                p { class: "muted", "The branch goes under a decision of the tree, such as dictate." }
                            }
                            pre { class: "code", "{draft}" }
                        }
                    }
                    _ => rsx! { p { class: "muted", "Capture an application in the Context tab first." } },
                }
                if let Some(message) = message() {
                    p { class: "ok-text", "{message}" }
                }
            }
        }

        div { class: "dx-accordion",
            {rows.into_iter().enumerate().map(|(i, row)| {
                let indent = "  ".repeat(row.depth);
                let title = format!("{indent}{}", row.title);
                let kind = format!("{:?}", row.kind).to_lowercase();
                rsx! {
                    Collapsible { key: "{i}-{row.path}", title, subtitle: Some(row.details.clone()), open: false,
                        div { class: "row",
                            {badge(&kind, "secondary")}
                            span { class: "muted mono grow", "{row.file}" }
                        }
                        if !row.description.is_empty() {
                            p { "{row.description}" }
                        }
                        pre { class: "code", "{row.text}" }
                    }
                }
            })}
        }
    }
}
