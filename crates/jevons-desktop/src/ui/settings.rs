//! Runtime, dictation and privacy settings, applied and saved together.

use super::Ctx;
use super::components::{Choice, HotkeyField, Select, Switch, badge, copy};
use crate::agent::Command;
use crate::runtime::Status;
use dioxus::prelude::*;
use jevons_desktop_core::config::{DesktopConfig, HotkeyMode, Mode};

/// The hotkey mode a switch sets: on holds the hotkey while speaking.
fn mode(hold: bool) -> HotkeyMode {
    if hold {
        HotkeyMode::Hold
    } else {
        HotkeyMode::Toggle
    }
}

fn mode_label(mode: HotkeyMode) -> String {
    match mode {
        HotkeyMode::Hold => "Hold while speaking; release to finish",
        HotkeyMode::Toggle => "Press to start, press again to finish",
    }
    .into()
}

/// The settings being edited, kept by the window rather than the page: edits survive switching
/// tabs, and the footer outside the scrolling page saves them. Until something is edited it
/// shows the saved settings, so changes made elsewhere (the tray, the Models tab) show through.
#[derive(Clone, Copy, PartialEq)]
pub struct SettingsDraft {
    /// The settings as saved, refreshed by the window on every render (not reactive).
    saved: CopyValue<DesktopConfig>,
    /// The edits, from the first change until they are applied or reverted.
    config: Signal<Option<DesktopConfig>>,
    /// The bind address as typed, which may not parse yet.
    bind: Signal<Option<String>>,
}

impl SettingsDraft {
    pub fn new(saved: DesktopConfig) -> Self {
        Self {
            saved: CopyValue::new(saved),
            config: Signal::new(None),
            bind: Signal::new(None),
        }
    }

    /// Takes the settings as they are saved now.
    pub fn follow(&self, saved: &DesktopConfig) {
        let mut cell = self.saved;
        if *cell.peek() != *saved {
            cell.set(saved.clone());
        }
    }

    /// The settings the page shows: the edits, or else the saved settings.
    pub fn current(&self) -> DesktopConfig {
        self.config
            .read()
            .clone()
            .unwrap_or_else(|| self.saved.peek().clone())
    }

    /// Changes the edited settings, starting the edits from the saved ones.
    pub fn edit(&self, change: impl FnOnce(&mut DesktopConfig)) {
        let saved = self.saved;
        let mut config = self.config;
        change(config.write().get_or_insert_with(|| saved.peek().clone()));
    }

    pub fn bind(&self) -> String {
        self.bind
            .read()
            .clone()
            .unwrap_or_else(|| self.saved.peek().server.bind.to_string())
    }

    pub fn set_bind(&self, text: String) {
        let mut bind = self.bind;
        bind.set(Some(text));
    }

    /// The settings to save, or `None` while the bind address does not parse. The model choices
    /// belong to the Models tab and approvals to their own gesture, so the saved ones are kept.
    pub fn saving(&self) -> Option<DesktopConfig> {
        let saved = self.saved.peek();
        let mut config = self.current();
        config.server.bind = self.bind().trim().parse().ok()?;
        config.models = saved.models.clone();
        config.automation.approved = saved.automation.approved.clone();
        Some(config)
    }

    /// Whether anything differs from the saved settings.
    pub fn dirty(&self) -> bool {
        let edited = self.config.read().is_some() || self.bind.read().is_some();
        edited
            && match self.saving() {
                Some(config) => config != *self.saved.peek(),
                None => true,
            }
    }

    /// Drops the edits.
    pub fn revert(&self) {
        let (mut config, mut bind) = (self.config, self.bind);
        config.set(None);
        bind.set(None);
    }
}

/// The bar under the page while settings are unsaved: what to do with them, always in view.
pub fn footer(ctx: &Ctx, draft: SettingsDraft) -> Element {
    let apply = ctx.clone();
    let valid = draft.saving().is_some();
    rsx! {
        div { class: "save-bar",
            span { class: "save-bar-dot" }
            span { class: "grow", "Unsaved settings" }
            if !valid {
                span { class: "error-text", "The bind address is not an IP address" }
            }
            button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                onclick: move |_| draft.revert(),
                "Revert"
            }
            button { class: "dx-button", "data-style": "accent", "data-size": "sm", disabled: !valid,
                onclick: move |_| {
                    if let Some(config) = draft.saving() {
                        apply.send(Command::Apply(Box::new(config)));
                        draft.revert();
                    }
                },
                "Apply and save"
            }
        }
    }
}

#[component]
pub fn SettingsPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    // The window's draft; a page shown on its own keeps one of its own.
    let draft = use_hook(|| {
        try_consume_context::<SettingsDraft>().unwrap_or_else(|| {
            SettingsDraft::new(ctx.view.lock().expect("the view lock").config.clone())
        })
    });
    let view = ctx.view.lock().expect("the view lock");
    let config_file = view.config_file.display().to_string();
    let status = view.runtime.clone();
    let devices = view.devices.clone();
    let entries = view.flows.entries();
    let automations = view.automations.clone();
    drop(view);

    let d = draft.current();
    let bind = draft.bind();
    let bind_ok = bind.trim().parse::<std::net::IpAddr>().is_ok();
    let dirty = draft.dirty();
    let open_api = d.server.expose && d.exposed_key().is_none();
    let mut microphones = vec![Choice {
        value: None,
        label: "Default microphone".into(),
    }];
    microphones.extend(devices.iter().map(|m| Choice {
        value: Some(m.name.clone()),
        label: m.name.clone(),
    }));

    rsx! {
        div { class: "spread",
            div { class: "row",
                h2 { "Settings" }
                if dirty {
                    {badge("Unsaved changes", "warning")}
                }
            }
            span { class: "muted mono", "{config_file}" }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Runtime" }
                    div { class: "dx-card-description",
                        {status.as_ref().map_or_else(|| "Starting…".to_string(), Status::describe)}
                    }
                }
                if let Some(Status::Ready { base_url, exposed: true }) = status.clone() {
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        onclick: move |_| copy(&format!("{base_url}/v1")), "Copy base URL" }
                }
            }
            div { class: "dx-card-content",
                div { class: "dx-tabs-list",
                    button { class: "dx-tabs-trigger",
                        "data-state": if d.server.mode == Mode::Embedded { "active" } else { "inactive" },
                        onclick: move |_| draft.edit(|c| c.server.mode = Mode::Embedded),
                        "Run the models in this app" }
                    button { class: "dx-tabs-trigger",
                        "data-state": if d.server.mode == Mode::Remote { "active" } else { "inactive" },
                        onclick: move |_| draft.edit(|c| c.server.mode = Mode::Remote),
                        "Use a jevons server" }
                }
                if d.server.mode == Mode::Embedded {
                    div { class: "field",
                        span { class: "field-label", "Expose the API" }
                        Switch { checked: d.server.expose, label: "Serve other clients (OpenAI SDK, scripts)".to_string(),
                            onchange: move |on| draft.edit(|c| c.server.expose = on) }
                    }
                    if d.server.expose {
                        div { class: "field",
                            span { class: "field-label", "Address and port" }
                            div { class: "row",
                                input { class: "dx-input mono", value: "{bind}", oninput: move |e| draft.set_bind(e.value()) }
                                input { class: "dx-input narrow mono", value: "{d.server.port}",
                                    oninput: move |e| if let Ok(port) = e.value().trim().parse() { draft.edit(|c| c.server.port = port) } }
                            }
                        }
                        if !bind_ok {
                            p { class: "error-text", "The address must be an IP address, such as 127.0.0.1 or 0.0.0.0." }
                        }
                        div { class: "field",
                            span { class: "field-label", "API key" }
                            input { class: "dx-input mono", placeholder: "or set TYPESAFE_API_KEY",
                                value: "{d.server.api_key.clone().unwrap_or_default()}",
                                oninput: move |e| draft.edit(|c| c.server.api_key = Some(e.value()).filter(|k| !k.is_empty())) }
                        }
                        if open_api {
                            p { class: "warn", "Without a key, anyone who can reach the port can use the API." }
                        }
                    }
                } else {
                    div { class: "field",
                        span { class: "field-label", "Server URL" }
                        input { class: "dx-input mono", value: "{d.server.remote_url}",
                            oninput: move |e| draft.edit(|c| c.server.remote_url = e.value()) }
                    }
                    div { class: "field",
                        span { class: "field-label", "API key" }
                        input { class: "dx-input mono", value: "{d.server.remote_key.clone().unwrap_or_default()}",
                            oninput: move |e| draft.edit(|c| c.server.remote_key = Some(e.value()).filter(|k| !k.is_empty())) }
                    }
                }
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Dictation" }
                    div { class: "dx-card-description", "Click a hotkey field, then press the combination" }
                }
            }
            div { class: "dx-card-content",
                div { class: "field",
                    div { class: "stack",
                        span { class: "field-label", "Push-to-talk" }
                        span { class: "field-hint", "Also the branch hotkeys" }
                    }
                    HotkeyField { value: d.dictation.hotkey.clone(), optional: false,
                        onchange: move |h: String| if !h.is_empty() { draft.edit(|c| c.dictation.hotkey = h) } }
                }
                div { class: "field",
                    span { class: "field-label", "" }
                    Switch { checked: d.dictation.hotkey_mode == HotkeyMode::Hold,
                        label: mode_label(d.dictation.hotkey_mode),
                        onchange: move |on| draft.edit(|c| c.dictation.hotkey_mode = mode(on)) }
                }
                div { class: "field",
                    div { class: "stack",
                        span { class: "field-label", "Live dictation" }
                        span { class: "field-hint", "Inserts the text when it stops" }
                    }
                    HotkeyField { value: d.dictation.live_hotkey.clone().unwrap_or_default(), optional: true,
                        onchange: move |h: String| draft.edit(|c| c.dictation.live_hotkey = Some(h).filter(|h| !h.is_empty())) }
                }
                div { class: "field",
                    span { class: "field-label", "" }
                    Switch { checked: d.dictation.live_hotkey_mode == HotkeyMode::Hold,
                        label: mode_label(d.dictation.live_hotkey_mode),
                        onchange: move |on| draft.edit(|c| c.dictation.live_hotkey_mode = mode(on)) }
                }
                div { class: "field",
                    span { class: "field-label", "Live feedback" }
                    Switch { checked: d.dictation.live_feedback,
                        label: "A bubble by the tray icon shows what dictation hears and does".to_string(),
                        onchange: move |on| draft.edit(|c| c.dictation.live_feedback = on) }
                }
                div { class: "field",
                    span { class: "field-label", "Show this window" }
                    HotkeyField { value: d.dictation.inspector_hotkey.clone().unwrap_or_default(), optional: true,
                        onchange: move |h: String| draft.edit(|c| c.dictation.inspector_hotkey = Some(h).filter(|h| !h.is_empty())) }
                }
                div { class: "field",
                    span { class: "field-label", "Microphone" }
                    Select { value: d.dictation.microphone.clone(), choices: microphones,
                        onchange: move |m| draft.edit(|c| c.dictation.microphone = m) }
                }
                div { class: "field",
                    span { class: "field-label", "Language" }
                    input { class: "dx-input narrow", placeholder: "detect",
                        value: "{d.dictation.language.clone().unwrap_or_default()}",
                        oninput: move |e| draft.edit(|c| c.dictation.language = Some(e.value().trim().to_lowercase()).filter(|l| !l.is_empty())) }
                }
                div { class: "field",
                    span { class: "field-label", "Decide" }
                    Switch { checked: d.dictation.decide, label: "Ask the decision model at the flow tree's decisions (off: their fallbacks)".to_string(),
                        onchange: move |on| draft.edit(|c| c.dictation.decide = on) }
                }
                div { class: "field",
                    span { class: "field-label", "Max output tokens" }
                    input { class: "dx-input narrow mono", value: "{d.dictation.max_output_tokens}",
                        oninput: move |e| if let Ok(n) = e.value().trim().parse::<u32>() { draft.edit(|c| c.dictation.max_output_tokens = n.clamp(16, 8192)) } }
                }
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Start at a branch" }
                    div { class: "dx-card-description", "A push-to-talk hotkey that starts the take at this branch of the flow tree instead of its root" }
                }
            }
            div { class: "dx-card-content",
                {entries.into_iter().map(|(id, description)| {
                    let value = d.dictation.branch_hotkeys.get(&id).cloned().unwrap_or_default();
                    let key = id.clone();
                    rsx! {
                        div { class: "field", key: "{id}",
                            div { class: "stack",
                                span { class: "field-label mono", "{id}" }
                                span { class: "field-hint", "{description}" }
                            }
                            HotkeyField { value, optional: true,
                                onchange: move |h: String| draft.edit(|c| {
                                    if h.is_empty() {
                                        c.dictation.branch_hotkeys.remove(&key);
                                    } else {
                                        c.dictation.branch_hotkeys.insert(key.clone(), h);
                                    }
                                }) }
                        }
                    }
                })}
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Automations" }
                    div { class: "dx-card-description", "Record a task once, then run it again from the tray, a hotkey or by saying it" }
                }
            }
            div { class: "dx-card-content",
                div { class: "field",
                    div { class: "stack",
                        span { class: "field-label", "Record" }
                        span { class: "field-hint", "Starts recording; while recording, hold it to say what the task is (or a note), tap it to finish" }
                    }
                    HotkeyField { value: d.automation.record_hotkey.clone().unwrap_or_default(), optional: true,
                        onchange: move |h: String| draft.edit(|c| c.automation.record_hotkey = Some(h).filter(|h| !h.is_empty())) }
                }
                {automations.into_iter().map(|(name, description, approved)| {
                    let value = d.automation.hotkeys.get(&name).cloned().unwrap_or_default();
                    let key = name.clone();
                    let hint = if approved { description } else { format!("{description} (not approved yet)") };
                    rsx! {
                        div { class: "field", key: "{name}",
                            div { class: "stack",
                                span { class: "field-label mono", "{name}" }
                                span { class: "field-hint", "{hint}" }
                            }
                            HotkeyField { value, optional: true,
                                onchange: move |h: String| draft.edit(|c| {
                                    if h.is_empty() {
                                        c.automation.hotkeys.remove(&key);
                                    } else {
                                        c.automation.hotkeys.insert(key.clone(), h);
                                    }
                                }) }
                        }
                    }
                })}
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Privacy" }
                    div { class: "dx-card-description", "What the context keeps of your text" }
                }
            }
            div { class: "dx-card-content",
                div { class: "field",
                    span { class: "field-label", "Characters per field" }
                    input { class: "dx-input narrow mono", value: "{d.privacy.max_context_chars}",
                        oninput: move |e| if let Ok(n) = e.value().trim().parse::<usize>() { draft.edit(|c| c.privacy.max_context_chars = n.min(20_000)) } }
                }
                div { class: "field",
                    span { class: "field-label", "Clipboard" }
                    Switch { checked: d.privacy.read_clipboard, label: "Include the clipboard in the context".to_string(),
                        onchange: move |on| draft.edit(|c| c.privacy.read_clipboard = on) }
                }
                div { class: "field",
                    span { class: "field-label", "API log" }
                    Switch { checked: d.privacy.log_api,
                        label: "Write every decision and generation request and response to logs/api.log".to_string(),
                        onchange: move |on| draft.edit(|c| c.privacy.log_api = on) }
                }
                if d.privacy.log_api {
                    p { class: "warn", "The log holds what you say and the text of your screen, in full. Turn it off when you are done debugging." }
                }
            }
        }

    }
}
