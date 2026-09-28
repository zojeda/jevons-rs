//! The window: a top bar with the tabs, the page, and a status bar with the live text.

use super::components::badge;
use super::{Ctx, context, models, profiles, settings, takes};
use crate::runtime::Status;
use dioxus::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Context,
    Takes,
    Profiles,
    Settings,
    Models,
}

const TABS: [(Tab, &str); 5] = [
    (Tab::Context, "Context"),
    (Tab::Takes, "Takes"),
    (Tab::Profiles, "Profiles"),
    (Tab::Settings, "Settings"),
    (Tab::Models, "Models"),
];

#[component]
pub fn App() -> Element {
    let ctx = use_context::<Ctx>();
    let mut tab = use_signal(|| Tab::Context);
    let frozen = use_signal(|| false);
    // Pages read the shared view; a new revision on every render makes them re-render when the
    // agent wakes the window.
    let revision = use_hook(|| std::rc::Rc::new(std::cell::Cell::new(0u64)));
    revision.set(revision.get() + 1);
    let rev = revision.get();

    let mut view = ctx.view.lock().expect("the view lock");
    // The agent reads the focused window twice a second while the Context tab shows it.
    view.watch_context = tab() == Tab::Context && !frozen();
    let (status, status_style) = match &view.runtime {
        Some(status @ Status::Ready { .. }) | Some(status @ Status::Remote { .. }) => {
            (status.describe(), "success")
        }
        Some(status @ Status::Failed(_)) => (status.describe(), "destructive"),
        Some(status) => (status.describe(), "warning"),
        None => ("Starting…".to_string(), "warning"),
    };
    let tuning = crate::tuning::active();
    let tray = view
        .tray
        .tooltip()
        .trim_start_matches("jevons: ")
        .to_string();
    let hotkey = view.config.dictation.hotkey.clone();
    let live_hotkey = view.config.dictation.live_hotkey.clone();
    let hotkey_error = view.hotkey_error.clone();
    let notice = view.notice.clone();
    let hearing =
        (view.dictating || !view.live_transcript.is_empty()).then(|| view.live_transcript.clone());
    let writing = (!view.live_output.is_empty()).then(|| view.live_output.clone());
    drop(view);

    rsx! {
        div { class: "app",
            div { class: "topbar",
                div { class: "brand",
                    span { class: "brand-name", "jevons" }
                    {badge(&status, status_style)}
                    if tuning {
                        {badge("Tuning GPU kernels (first runs only)", "warning")}
                    }
                }
                div { class: "dx-tabs-list",
                    {TABS.iter().map(|&(t, label)| rsx! {
                        button {
                            class: "dx-tabs-trigger",
                            "data-state": if tab() == t { "active" } else { "inactive" },
                            onclick: move |_| tab.set(t),
                            "{label}"
                        }
                    })}
                }
            }
            div { class: "page",
                match tab() {
                    Tab::Context => rsx! { context::ContextPage { rev, frozen } },
                    Tab::Takes => rsx! { takes::TakesPage { rev } },
                    Tab::Profiles => rsx! { profiles::ProfilesPage { rev } },
                    Tab::Settings => rsx! { settings::SettingsPage { rev } },
                    Tab::Models => rsx! { models::ModelsPage { rev } },
                }
            }
            if let Some(text) = hearing {
                div { class: "live", strong { "Hearing" } "{text}" }
            }
            if let Some(text) = writing {
                div { class: "live", strong { "Writing" } "{text}" }
            }
            div { class: "statusbar",
                span { "{tray}" }
                span { "·" }
                span { class: "mono", "hold {hotkey}" }
                if let Some(live) = live_hotkey {
                    span { class: "mono", "· live {live}" }
                }
                if let Some(error) = hotkey_error {
                    span { class: "error-text", "{error}" }
                }
                if let Some(notice) = notice {
                    span { class: "warn", "· {notice}" }
                }
            }
        }
    }
}
