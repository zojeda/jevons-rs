//! The live context and the route it takes through the flow tree.

use super::Ctx;
use super::components::{Collapsible, Icon, JsonTree, Switch, badge, icon};
use crate::agent::Command;
use dioxus::prelude::*;
use jevons_desktop_core::flow::Check;
use jevons_desktop_core::flow::walk::FlowStep;
use std::time::Duration;

#[component]
pub fn ContextPage(rev: u64, frozen: Signal<bool>) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let mut raw = use_signal(|| false);
    let view = ctx.view.lock().expect("the view lock");
    let backends = format!(
        "context: {} · delivery: {}",
        view.context_backend, view.sink_backend
    );
    let paused = view.context_paused;
    let error = view.context_error.clone();
    let context = view.context.clone();
    let route = view.route.clone();
    drop(view);
    let capture = ctx.clone();
    let record = ctx.clone();

    rsx! {
        div { class: "spread",
            div { class: "row",
                Switch {
                    checked: frozen(),
                    label: "Freeze".to_string(),
                    onchange: move |on| frozen.set(on),
                }
                button {
                    class: "dx-button",
                    "data-style": "outline",
                    "data-size": "sm",
                    onclick: move |_| {
                        frozen.set(true);
                        capture.send(Command::CaptureContextIn(Duration::from_secs(3)));
                    },
                    "Capture in 3 s"
                }
                Switch {
                    checked: raw(),
                    label: "Raw JSON".to_string(),
                    onchange: move |on| raw.set(on),
                }
                button {
                    class: "dx-button",
                    "data-style": "outline",
                    "data-size": "sm",
                    title: "Save this window's interface to ~/jevons/trees, for writing investigations and replaying them with --tree",
                    onclick: move |_| record.send(Command::RecordTree),
                    "Record tree"
                }
            }
            span { class: "muted", "{backends}" }
        }
        if paused {
            p { class: "warn", "Context capture is paused (tray menu)." }
        }
        if let Some(error) = error {
            p { class: "error-text", "{error}" }
        }
        match context {
            None => rsx! {
                div { class: "dx-card",
                    div { class: "dx-card-content",
                        p { class: "muted",
                            "Switch to another application: the focused window's context appears here. \
                             Use Capture in 3 s to switch to the application you want to inspect."
                        }
                    }
                }
            },
            Some(context) => {
                let value = serde_json::to_value(&context).unwrap_or_default();
                let mut rows: Vec<(String, String)> = vec![
                    ("Application".into(), context.app.process_name.clone()),
                    ("Window".into(), context.window.title.clone()),
                ];
                if let Some(url) = &context.url {
                    rows.push(("Address".into(), url.clone()));
                }
                if let Some(e) = &context.focused {
                    rows.push(("Role".into(), e.role.clone()));
                    rows.push(("Name".into(), e.name.clone()));
                    if let Some(id) = &e.automation_id {
                        rows.push(("Automation id".into(), id.clone()));
                    }
                    rows.push(("Editable".into(), if e.is_editable { "yes" } else { "no" }.into()));
                    if e.is_password {
                        rows.push(("Password".into(), "yes (its text is never read)".into()));
                    }
                    for (key, text) in [
                        ("Selection", &e.selection),
                        ("Before the caret", &e.before_caret),
                        ("After the caret", &e.after_caret),
                        ("Value", &e.value_excerpt),
                    ] {
                        if let Some(text) = text {
                            rows.push((key.into(), excerpt(text)));
                        }
                    }
                }
                for (key, text) in &context.extras {
                    rows.push((key.clone(), excerpt(text)));
                }
                let json = serde_json::to_string_pretty(&value).unwrap_or_default();
                rsx! {
                    div { class: "dx-card",
                        div { class: "dx-card-header",
                            div {
                                div { class: "dx-card-title", "Focused window" }
                                div { class: "dx-card-description", "What the accessibility layer reports right now" }
                            }
                        }
                        div { class: "dx-card-content",
                            if raw() {
                                pre { class: "code", "{json}" }
                            } else {
                                div { class: "kv",
                                    {rows.into_iter().map(|(k, v)| rsx! {
                                        span { class: "k", "{k}" }
                                        span { class: "v", "{v}" }
                                    })}
                                }
                                if !context.errors.is_empty() {
                                    div { class: "stack",
                                        {context.errors.iter().map(|e| rsx! { p { class: "warn", "{e}" } })}
                                    }
                                }
                                Collapsible { title: "Snapshot as a tree".to_string(), subtitle: None, open: false,
                                    JsonTree { value }
                                }
                            }
                        }
                    }
                }
            }
        }
        if !route.is_empty() {
            {route_card(&route)}
        }
    }
}

fn excerpt(text: &str) -> String {
    let short: String = text.chars().take(300).collect();
    if short.len() < text.len() {
        format!("{short}…")
    } else {
        short
    }
}

/// The route the context takes before any model decision, with every guard checked.
pub fn route_card(route: &[FlowStep]) -> Element {
    let path: Vec<String> = route.iter().filter_map(|s| s.chosen.clone()).collect();
    let end = route.last().and_then(|s| s.how.clone());
    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Route" }
                    div { class: "dx-card-description", "Where a take from here goes, by guards and rules alone, with each rule checked" }
                }
                div { class: "row",
                    {badge(&if path.is_empty() { "/".to_string() } else { path.join(" / ") }, "accent")}
                }
            }
            div { class: "dx-card-content",
                if let Some(end) = end {
                    p { class: "muted", "{end}" }
                }
                div { class: "dx-accordion",
                    {route.iter().enumerate().map(|(i, step)| {
                        let title = step.node.clone();
                        let subtitle = match (&step.chosen, &step.how) {
                            (Some(chosen), Some(how)) => format!("→ {chosen} ({how})"),
                            (None, Some(how)) => how.clone(),
                            _ => format!("{:?}", step.kind).to_lowercase(),
                        };
                        let open = i + 1 == route.len();
                        rsx! {
                            Collapsible { key: "{step.node}", title, subtitle: Some(subtitle), open,
                                if step.branches.is_empty() {
                                    p { class: "muted", "A leaf: the walk ends here." }
                                }
                                {step.branches.iter().map(|b| {
                                    rsx! {
                                        div { class: "row",
                                            {icon(if b.passed { Icon::Check } else { Icon::Cross })}
                                            span { class: "mono", "{b.name}" }
                                            span { class: "muted", "priority {b.priority} · {b.specificity} rules" }
                                        }
                                        {checks(&b.checks)}
                                    }
                                })}
                            }
                        }
                    })}
                }
            }
        }
    }
}

fn checks(checks: &[Check]) -> Element {
    if checks.is_empty() {
        return rsx! { p { class: "muted", "No guard: always applies." } };
    }
    rsx! {
        div {
            {checks.iter().map(|check| {
                let mark = icon(if check.passed { Icon::Check } else { Icon::Cross });
                let value = check.value.clone().unwrap_or_else(|| "(nothing)".into());
                rsx! {
                    div { class: "check",
                        span { {mark} }
                        span { class: "mono", "{check.rule}" }
                        span { class: "mono", "{check.pattern}" }
                        span { "{value}" }
                    }
                }
            })}
        }
    }
}
