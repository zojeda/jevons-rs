//! The live context and the profile resolution for it.

use super::Ctx;
use super::components::{Collapsible, JsonTree, Switch, badge};
use crate::agent::Command;
use dioxus::prelude::*;
use jevons_desktop_core::profile::{Check, Resolution};
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
    let resolution = view.resolution.clone();
    drop(view);
    let capture = ctx.clone();

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
        if let Some(resolution) = resolution {
            {resolution_card(&resolution)}
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

/// Which profile and destination win, and every rule checked.
pub fn resolution_card(resolution: &Resolution) -> Element {
    let winner = resolution.profile.clone();
    let destination = resolution.destination.clone();
    let tied = (!resolution.tied.is_empty()).then(|| resolution.tied.join(", "));
    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Profile" }
                    div { class: "dx-card-description", "Every profile, best first, with each rule it checked" }
                }
                div { class: "row",
                    {badge(&winner, "accent")}
                    if let Some(destination) = destination {
                        {badge(&format!("→ {destination}"), "secondary")}
                    }
                    if resolution.forced {
                        {badge("forced from the tray", "warning")}
                    }
                }
            }
            div { class: "dx-card-content",
                if let Some(tied) = tied {
                    p { class: "muted", "Tied with {tied}: the decision model chooses." }
                }
                div { class: "dx-accordion",
                    {resolution.trace.iter().map(|profile| {
                        let mark = if profile.matched { "✔" } else { "✘" };
                        let priority = if profile.priority == i32::MIN {
                            "lowest".to_string()
                        } else {
                            profile.priority.to_string()
                        };
                        let title = format!("{mark} {}", profile.id);
                        let subtitle = format!("priority {priority} · {} rules", profile.specificity);
                        let open = profile.id == resolution.profile;
                        rsx! {
                            Collapsible { key: "{profile.id}", title, subtitle: Some(subtitle), open,
                                {checks(&profile.checks)}
                                {profile.destinations.iter().map(|d| {
                                    let mark = if d.matched { "✔" } else { "✘" };
                                    rsx! {
                                        p { class: "muted", "{mark} destination {d.id} · priority {d.priority}" }
                                        {checks(&d.checks)}
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
        return rsx! { p { class: "muted", "No rules: matches everything." } };
    }
    rsx! {
        div {
            {checks.iter().map(|check| {
                let (mark, class) = if check.passed { ("✔", "pass") } else { ("✘", "fail") };
                let value = check.value.clone().unwrap_or_else(|| "(nothing)".into());
                rsx! {
                    div { class: "check",
                        span { class: "{class}", "{mark}" }
                        span { class: "mono", "{check.rule}" }
                        span { class: "mono", "{check.pattern}" }
                        span { "{value}" }
                    }
                }
            })}
        }
    }
}
