//! The window: a top bar with the tabs, the page, and a status bar with the live text.

use super::components::badge;
use super::{Ctx, context, flows, machines, models, settings, takes};
use crate::runtime::Status;
use dioxus::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Context,
    Takes,
    Flows,
    Machines,
    Settings,
    Models,
}

const TABS: [(Tab, &str); 6] = [
    (Tab::Context, "Context"),
    (Tab::Takes, "Takes"),
    (Tab::Flows, "Flows"),
    (Tab::Machines, "Machines"),
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

    let initial = ctx.view.lock().expect("the view lock").config.clone();
    // The settings being edited live here, not in the page: they survive switching tabs, and the
    // save bar outside the scrolling page saves them.
    let draft = use_context_provider(|| settings::SettingsDraft::new(initial));

    let mut view = ctx.view.lock().expect("the view lock");
    draft.follow(&view.config);
    // The agent reads the focused window twice a second while the Context tab shows it.
    view.watch_context = tab() == Tab::Context && !frozen();
    let (label, status, status_style) = match &view.runtime {
        Some(status @ Status::Ready { .. }) | Some(status @ Status::Remote { .. }) => {
            (status.label(), status.describe(), "success")
        }
        Some(status @ Status::Failed(_)) => (status.label(), status.describe(), "destructive"),
        Some(status) => (status.label(), status.describe(), "warning"),
        None => ("Starting", "Starting…".to_string(), "warning"),
    };
    // Without models, or when they failed, the badge leads to the Models tab.
    let to_models = matches!(view.runtime, Some(Status::NoModels | Status::Failed(_)));
    let tuning = crate::tuning::active();
    let tray = view
        .tray
        .tooltip()
        .trim_start_matches("jevons: ")
        .to_string();
    let verb = |mode| match mode {
        jevons_desktop_core::config::HotkeyMode::Hold => "hold",
        jevons_desktop_core::config::HotkeyMode::Toggle => "press",
    };
    let hotkey = format!(
        "{} {}",
        verb(view.config.dictation.hotkey_mode),
        view.config.dictation.hotkey
    );
    let live_hotkey = view
        .config
        .dictation
        .live_hotkey
        .clone()
        .map(|live| format!("{} {live}", verb(view.config.dictation.live_hotkey_mode)));
    let dictating = view.dictating;
    let hotkey_error = view.hotkey_error.clone();
    let notice = view.notice.clone();
    let hearing =
        (view.dictating || !view.live_transcript.is_empty()).then(|| view.live_transcript.clone());
    let writing = (!view.live_output.is_empty()).then(|| view.live_output.clone());
    drop(view);
    let unsaved = draft.dirty();

    rsx! {
        div { class: "app",
            div { class: "topbar",
                div { class: "brand",
                    span { class: "brand-name", "jevons" }
                    if to_models {
                        button { class: "badge-button", onclick: move |_| tab.set(Tab::Models),
                            {badge(&format!("{label}: open Models"), status_style)}
                        }
                    } else {
                        {badge(label, status_style)}
                    }
                    if tuning {
                        {badge("Tuning GPU kernels", "warning")}
                    }
                }
                div { class: "dx-tabs-list",
                    {TABS.iter().map(|&(t, label)| rsx! {
                        button {
                            class: "dx-tabs-trigger",
                            "data-state": if tab() == t { "active" } else { "inactive" },
                            onclick: move |_| tab.set(t),
                            "{label}"
                            if t == Tab::Settings && unsaved {
                                span { class: "tab-dot", title: "Unsaved settings" }
                            }
                        }
                    })}
                }
            }
            div { class: "page",
                "data-dirty": if tab() == Tab::Settings && unsaved { "true" } else { "false" },
                match tab() {
                    Tab::Context => rsx! { context::ContextPage { rev, frozen } },
                    Tab::Takes => rsx! { takes::TakesPage { rev } },
                    Tab::Flows => rsx! { flows::FlowsPage { rev } },
                    Tab::Machines => rsx! { machines::MachinesPage { rev } },
                    Tab::Settings => rsx! { settings::SettingsPage { rev } },
                    Tab::Models => rsx! { models::ModelsPage { rev } },
                }
            }
            if tab() == Tab::Settings && unsaved {
                {settings::footer(&ctx, draft)}
            }
            if let Some(text) = hearing {
                div { class: "live", strong { if dictating { "Hearing" } else { "Last heard" } } "{text}" }
            }
            if let Some(text) = writing {
                div { class: "live", strong { if dictating { "Writing" } else { "Last written" } } "{text}" }
            }
            div { class: "statusbar",
                span { "{status}" }
                span { "·" }
                span { "{tray}" }
                span { "·" }
                span { class: "mono", "{hotkey}" }
                if let Some(live) = live_hotkey {
                    span { class: "mono", "· live: {live}" }
                }
                if let Some(error) = hotkey_error {
                    span { "·" }
                    span { class: "error-text", "{error}" }
                }
                if let Some(notice) = notice {
                    span { class: "warn", "· {notice}" }
                }
            }
        }
    }
}
