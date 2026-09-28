//! The loaded profiles, their errors, and new profiles drafted from the current context.

use super::Ctx;
use super::components::{Collapsible, badge};
use crate::agent::{Command, open_folder};
use dioxus::prelude::*;
use jevons_desktop_core::profile::draft;

#[component]
pub fn ProfilesPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let mut new_id = use_signal(String::new);
    let mut message = use_signal(|| None::<String>);
    let view = ctx.view.lock().expect("the view lock");
    let dir = view.config.profiles_dir(&view.config_file);
    let errors: Vec<String> = view
        .profiles
        .errors
        .iter()
        .map(|e| format!("{}: {}", e.file.display(), e.message))
        .collect();
    let profiles: Vec<(String, String, i32, Option<std::path::PathBuf>, String)> = view
        .profiles
        .iter()
        .map(|p| {
            (
                p.spec.id.clone(),
                p.display_name().to_string(),
                p.spec.priority,
                p.source.clone(),
                toml::to_string_pretty(&p.spec).unwrap_or_default(),
            )
        })
        .collect();
    let context = view.context.clone();
    drop(view);

    let id = new_id();
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    let preview = context
        .as_ref()
        .map(|c| draft(c, if id.is_empty() { "new" } else { &id }));
    let open_dir = dir.clone();
    let reload = ctx.clone();

    rsx! {
        div { class: "spread",
            div { class: "stack",
                h2 { "Profiles" }
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
                    onclick: move |_| reload.send(Command::ReloadProfiles),
                    "Reload"
                }
            }
        }
        {errors.iter().map(|e| rsx! { p { class: "error-text", "{e}" } })}

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "New profile from the current context" }
                    div { class: "dx-card-description", "Its rules match the application, page and field the Context tab shows" }
                }
            }
            div { class: "dx-card-content",
                match (context, preview) {
                    (Some(context), Some(preview)) => {
                        let create_ctx = ctx.clone();
                        let dir = dir.clone();
                        rsx! {
                            p { class: "muted", "Matches {context.app.process_name} · {context.window.title}" }
                            div { class: "row",
                                input { class: "dx-input", placeholder: "profile id, such as slack", value: "{id}",
                                    oninput: move |e| new_id.set(e.value()) }
                                button { class: "dx-button", "data-style": "accent", "data-size": "sm", disabled: !valid,
                                    onclick: move |_| {
                                        let id = new_id();
                                        let file = dir.join(format!("{id}.toml"));
                                        message.set(Some(if file.exists() {
                                            format!("{} already exists", file.display())
                                        } else {
                                            let _ = std::fs::create_dir_all(&dir);
                                            match std::fs::write(&file, draft(&context, &id)) {
                                                Ok(()) => {
                                                    create_ctx.send(Command::ReloadProfiles);
                                                    open_folder(&file);
                                                    format!("Created {}; add its instructions and adjust its rules", file.display())
                                                }
                                                Err(e) => e.to_string(),
                                            }
                                        }));
                                    },
                                    "Create"
                                }
                            }
                            pre { class: "code", "{preview}" }
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
            {profiles.into_iter().map(|(id, name, priority, source, text)| {
                let priority = if priority == i32::MIN { "lowest".to_string() } else { priority.to_string() };
                let subtitle = format!("{id} · priority {priority}");
                rsx! {
                    Collapsible { key: "{id}", title: name, subtitle: Some(subtitle), open: false,
                        match source {
                            Some(file) => {
                                let shown = file.display().to_string();
                                rsx! {
                                    div { class: "row",
                                        span { class: "muted mono grow", "{shown}" }
                                        button { class: "dx-button", "data-style": "outline", "data-size": "xs",
                                            onclick: move |_| open_folder(&file), "Open" }
                                    }
                                }
                            }
                            None => rsx! { div { class: "row", {badge("built in", "secondary")} span { class: "muted", "Matches everything, at the lowest priority." } } },
                        }
                        pre { class: "code", "{text}" }
                    }
                }
            })}
        }
    }
}
