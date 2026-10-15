//! Runtime, dictation and privacy settings, applied and saved together.

use super::Ctx;
use super::components::{Choice, CopyButton, HotkeyField, Select, Switch, badge};
use crate::agent::Command;
use crate::config::{
    Capability, DesktopConfig, EMBEDDED, HotkeyMode, Provider, ProviderKind, RouteTo,
};
use crate::runtime::{Status, Voices};
use dioxus::prelude::*;
use std::collections::BTreeMap;

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

fn route_label(capability: Capability) -> &'static str {
    match capability {
        Capability::Speech => "Speech",
        Capability::Realtime => "Live speech",
        Capability::Decision => "Decisions",
        Capability::Generation => "Generation",
        Capability::Voice => "Voice",
    }
}

fn kind_label(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Embedded => "The models in this app",
        ProviderKind::Jevons => "A jevons server",
        ProviderKind::Openrouter => "OpenRouter",
        ProviderKind::OpenaiCompatible => "OpenAI-compatible",
    }
}

/// Sends `capability` to `provider`, with the provider's own model: a model's name belongs to
/// the provider it was asked of. `None` is the route left out (Realtime follows speech).
fn route_to(config: &mut DesktopConfig, capability: Capability, provider: Option<String>) {
    let route = provider.map(|provider| RouteTo {
        provider,
        model: None,
        voice: None,
    });
    set_route(config, capability, route);
}

/// Names the model `capability` asks its provider for; empty is the provider's own.
fn route_model(config: &mut DesktopConfig, capability: Capability, model: &str) {
    let Some(mut route) = config.route(capability) else {
        return;
    };
    route.model = Some(model.trim().to_string()).filter(|m| !m.is_empty());
    set_route(config, capability, Some(route));
}

/// Names the voice the voice route asks its provider to speak in; empty is the provider's own.
fn route_voice(config: &mut DesktopConfig, voice: &str) {
    let Some(mut route) = config.route(Capability::Voice) else {
        return;
    };
    route.voice = Some(voice.trim().to_string()).filter(|v| !v.is_empty());
    set_route(config, Capability::Voice, Some(route));
}

/// The voices to choose among for the voice route, when its provider lists them: the model's
/// own first, then each by name. A voice the settings name that the provider does not list
/// still shows as the choice. `None` when the provider lists none, and the name is typed.
fn voice_choices(listed: Option<&Voices>, named: Option<&str>) -> Option<Vec<Choice>> {
    let listed = listed.filter(|voices| !voices.names.is_empty())?;
    let mut choices = vec![Choice {
        value: None,
        label: format!("The model's own ({})", listed.default),
    }];
    choices.extend(listed.names.iter().map(|name| Choice {
        value: Some(name.clone()),
        label: name.clone(),
    }));
    if let Some(named) = named.filter(|n| listed.names.iter().all(|name| name != n)) {
        choices.push(Choice {
            value: Some(named.to_string()),
            label: format!("{named} (not listed)"),
        });
    }
    Some(choices)
}

/// Sets a route, leaving it out of the file when it is what a route left out means.
fn set_route(config: &mut DesktopConfig, capability: Capability, route: Option<RouteTo>) {
    config.routes.set(capability, None);
    if route != config.route(capability) {
        config.routes.set(capability, route);
    }
}

/// Adds a provider of `kind` under a name not taken yet.
fn add_provider(config: &mut DesktopConfig, kind: ProviderKind) {
    let base = match kind {
        ProviderKind::Embedded => EMBEDDED,
        ProviderKind::Jevons => "jevons",
        ProviderKind::Openrouter => "openrouter",
        ProviderKind::OpenaiCompatible => "openai",
    };
    let name = (1..)
        .map(|n| match n {
            1 => base.to_string(),
            n => format!("{base}-{n}"),
        })
        .find(|name| !config.providers.contains_key(name))
        .expect("a name is free");
    config.providers.insert(name, Provider::of(kind));
}

/// Removes a provider; the capabilities routed to it go back to the default.
fn remove_provider(config: &mut DesktopConfig, name: &str) {
    config.providers.remove(name);
    for capability in Capability::ALL {
        if config
            .routes
            .get(capability)
            .is_some_and(|route| route.provider == name)
        {
            config.routes.set(capability, None);
        }
    }
}

/// A setting typed as text, kept as typed and checked when saved: a number field rewritten on
/// every keystroke could not be cleared or typed digit by digit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Field {
    Bind,
    Port,
    Tokens,
    Chars,
}

/// The settings being edited, kept by the window rather than the page: edits survive switching
/// tabs, and the save bar outside the scrolling page saves them. Until something is edited it
/// shows the saved settings, so changes made elsewhere (the tray, the Models tab, approvals)
/// show through, and saving applies only what was edited onto the settings as they are then.
#[derive(Clone, Copy, PartialEq)]
pub struct SettingsDraft {
    /// The settings as saved, refreshed by the window on every render (not reactive).
    saved: CopyValue<DesktopConfig>,
    /// The saved settings the edits started from, and the edited ones.
    config: Signal<Option<(DesktopConfig, DesktopConfig)>>,
    /// Fields as typed.
    typed: Signal<BTreeMap<Field, String>>,
}

impl SettingsDraft {
    pub fn new(saved: DesktopConfig) -> Self {
        Self {
            saved: CopyValue::new(saved),
            config: Signal::new(None),
            typed: Signal::new(BTreeMap::new()),
        }
    }

    /// Takes the settings as they are saved now.
    pub fn follow(&self, saved: &DesktopConfig) {
        let mut cell = self.saved;
        if *cell.peek() != *saved {
            cell.set(saved.clone());
        }
    }

    /// The settings the page shows: the edited ones, or else the saved ones.
    pub fn current(&self) -> DesktopConfig {
        match &*self.config.read() {
            Some((_, edited)) => edited.clone(),
            None => self.saved.peek().clone(),
        }
    }

    /// Changes the edited settings, starting the edits from the saved ones.
    pub fn edit(&self, change: impl FnOnce(&mut DesktopConfig)) {
        let saved = self.saved;
        let mut config = self.config;
        let mut edits = config.write();
        let (_, edited) = edits.get_or_insert_with(|| (saved.peek().clone(), saved.peek().clone()));
        change(edited);
    }

    /// A typed field: as typed, or else its current value.
    pub fn typed(&self, field: Field) -> String {
        if let Some(text) = self.typed.read().get(&field) {
            return text.clone();
        }
        let c = self.current();
        match field {
            Field::Bind => c.server.bind.to_string(),
            Field::Port => c.server.port.to_string(),
            Field::Tokens => c.dictation.max_output_tokens.to_string(),
            Field::Chars => c.privacy.max_context_chars.to_string(),
        }
    }

    pub fn set_typed(&self, field: Field, text: String) {
        let mut typed = self.typed;
        typed.write().insert(field, text);
    }

    /// The settings to save: what was edited applied onto the settings as saved now, the typed
    /// fields checked. An error says which field is wrong.
    pub fn saving(&self) -> Result<DesktopConfig, String> {
        let saved = self.saved.peek().clone();
        let mut config = match &*self.config.read() {
            Some((base, edited)) => merge(base, edited, &saved),
            None => saved,
        };
        for (field, text) in self.typed.read().iter() {
            let text = text.trim();
            match field {
                Field::Bind => {
                    config.server.bind = text.parse().map_err(|_| {
                        "The address must be an IP address, such as 127.0.0.1 or 0.0.0.0"
                            .to_string()
                    })?
                }
                Field::Port => {
                    config.server.port = text
                        .parse()
                        .ok()
                        .filter(|p| *p > 0)
                        .ok_or("The port must be a number from 1 to 65535")?
                }
                Field::Tokens => {
                    config.dictation.max_output_tokens = text
                        .parse()
                        .ok()
                        .filter(|n| (16..=8192).contains(n))
                        .ok_or("Max output tokens must be from 16 to 8192")?
                }
                Field::Chars => {
                    config.privacy.max_context_chars = text
                        .parse()
                        .ok()
                        .filter(|n| (100..=20_000).contains(n))
                        .ok_or("Characters per field must be from 100 to 20000")?
                }
            }
        }
        config.check_routes()?;
        Ok(config)
    }

    /// Whether anything differs from the saved settings (or a typed field is wrong).
    pub fn dirty(&self) -> bool {
        let edited = self.config.read().is_some() || !self.typed.read().is_empty();
        edited && self.saving().ok().as_ref() != Some(&*self.saved.peek())
    }

    /// Drops the edits.
    pub fn revert(&self) {
        let (mut config, mut typed) = (self.config, self.typed);
        config.set(None);
        typed.set(BTreeMap::new());
    }
}

/// `edited`'s changes from `base`, applied onto `latest`, field by field: what changed elsewhere
/// meanwhile (a tray toggle, an approval, a model choice) is kept.
fn merge(base: &DesktopConfig, edited: &DesktopConfig, latest: &DesktopConfig) -> DesktopConfig {
    use serde_json::Value;
    fn apply(base: &Value, edited: &Value, latest: &mut Value) {
        match (base, edited, latest) {
            (Value::Object(base), Value::Object(edited), Value::Object(latest)) => {
                for (key, value) in edited {
                    let before = base.get(key).unwrap_or(&Value::Null);
                    if value == before {
                        continue;
                    }
                    match latest.get_mut(key) {
                        Some(now) if value.is_object() && before.is_object() => {
                            apply(before, value, now)
                        }
                        _ => {
                            latest.insert(key.clone(), value.clone());
                        }
                    }
                }
                // A setting the edits cleared (its key left out).
                for key in base.keys().filter(|k| !edited.contains_key(*k)) {
                    latest.remove(key);
                }
            }
            (base, edited, latest) if base != edited => *latest = edited.clone(),
            _ => {}
        }
    }
    let to_value = |c: &DesktopConfig| serde_json::to_value(c).unwrap_or_default();
    let mut merged = to_value(latest);
    apply(&to_value(base), &to_value(edited), &mut merged);
    serde_json::from_value(merged).unwrap_or_else(|_| edited.clone())
}

/// The bar under the page while settings are unsaved: what to do with them, always in view.
pub fn footer(ctx: &Ctx, draft: SettingsDraft) -> Element {
    let apply = ctx.clone();
    let problem = draft.saving().err();
    rsx! {
        div { class: "save-bar",
            span { class: "save-bar-dot" }
            span { class: "grow", "Unsaved settings" }
            if let Some(problem) = &problem {
                span { class: "error-text", "{problem}" }
            }
            button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                onclick: move |_| draft.revert(),
                "Revert"
            }
            button { class: "dx-button", "data-style": "accent", "data-size": "sm", disabled: problem.is_some(),
                onclick: move |_| {
                    if let Ok(config) = draft.saving() {
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
    let served = view.served.clone();
    let devices = view.devices.clone();
    let speakers = view.speakers.clone();
    let listed = view.voices.clone();
    let serving_voice = view.config.route(Capability::Voice).map(|r| r.provider);
    let entries = view.flows.entries();
    let automations = view.automations.clone();
    drop(view);

    let d = draft.current();
    let dirty = draft.dirty();
    let (bind, port) = (draft.typed(Field::Bind), draft.typed(Field::Port));
    let (tokens, chars) = (draft.typed(Field::Tokens), draft.typed(Field::Chars));
    let open_api = d.server.expose && d.exposed_key().is_none();
    let providers: Vec<(String, Provider)> = d
        .providers
        .iter()
        .map(|(name, provider)| (name.clone(), provider.clone()))
        .collect();
    let mut provider_choices = vec![Choice {
        value: Some(EMBEDDED.into()),
        label: "embedded (this app)".into(),
    }];
    provider_choices.extend(
        d.providers
            .keys()
            .filter(|n| *n != EMBEDDED)
            .map(|name| Choice {
                value: Some(name.clone()),
                label: name.clone(),
            }),
    );
    let mut microphones = vec![Choice {
        value: None,
        label: "Default microphone".into(),
    }];
    microphones.extend(devices.iter().map(|m| Choice {
        value: Some(m.name.clone()),
        label: m.name.clone(),
    }));
    let mut outputs = vec![Choice {
        value: None,
        label: "Default output device".into(),
    }];
    outputs.extend(speakers.iter().map(|s| Choice {
        value: Some(s.name.clone()),
        label: s.name.clone(),
    }));
    // A device the settings name that is not plugged in now still shows as the choice.
    if let Some(named) = d
        .voice
        .output
        .clone()
        .filter(|n| speakers.iter().all(|s| s.name != *n))
    {
        outputs.push(Choice {
            label: format!("{named} (not connected)"),
            value: Some(named),
        });
    }
    let voice_route = d.route(Capability::Voice);
    let voice_name = voice_route
        .as_ref()
        .and_then(|r| r.voice.clone())
        .unwrap_or_default();
    // Where the text of a spoken answer goes when the voice is not this app's own.
    let voice_elsewhere = voice_route
        .as_ref()
        .filter(|r| {
            d.provider(&r.provider)
                .is_some_and(|p| p.kind != ProviderKind::Embedded)
        })
        .map(|r| r.provider.clone());
    let voice_served = served.contains_key(Capability::Voice.key());
    // The voices listed are those of the provider that serves the voice now. Another provider
    // chosen on this page, and not saved yet, names its own.
    let voices = listed
        .as_ref()
        .filter(|_| voice_route.as_ref().map(|r| &r.provider) == serving_voice.as_ref());
    let voice_choices = voice_choices(voices, Some(voice_name.as_str()).filter(|n| !n.is_empty()));
    let voice_chosen = Some(voice_name.clone()).filter(|n| !n.is_empty());

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
                if let Some(api) = status.as_ref().and_then(Status::api) {
                    CopyButton { text: format!("{api}/v1"), label: "Copy base URL".to_string() }
                }
            }
            div { class: "dx-card-content",
                div { class: "field",
                    span { class: "field-label", "Expose the API" }
                    Switch { checked: d.server.expose, label: "Serve other clients (OpenAI SDK, scripts)".to_string(),
                        onchange: move |on| draft.edit(|c| c.server.expose = on) }
                }
                if d.server.expose {
                    div { class: "field",
                        span { class: "field-label", "Address and port" }
                        div { class: "row",
                            input { class: "dx-input mono", value: "{bind}", oninput: move |e| draft.set_typed(Field::Bind, e.value()) }
                            input { class: "dx-input narrow mono", value: "{port}",
                                oninput: move |e| draft.set_typed(Field::Port, e.value()) }
                        }
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
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Providers and routes" }
                    div { class: "dx-card-description",
                        "Each capability goes to a provider: the models in this app, a jevons server, OpenRouter or another server with OpenAI's API"
                    }
                }
            }
            div { class: "dx-card-content",
                {Capability::ALL.into_iter().map(|capability| {
                    let route = d.route(capability);
                    // Realtime left out follows speech, which the select says as such.
                    let provider = match capability {
                        Capability::Realtime => d.routes.realtime.as_ref().map(|r| r.provider.clone()),
                        _ => route.as_ref().map(|r| r.provider.clone()),
                    };
                    let model = route.as_ref().and_then(|r| r.model.clone()).unwrap_or_default();
                    let now = served.get(capability.key()).cloned().unwrap_or_else(|| "not served".into());
                    let mut choices = provider_choices.clone();
                    if capability == Capability::Realtime {
                        choices.insert(0, Choice { value: None, label: "follow speech".into() });
                    }
                    rsx! {
                        div { class: "field", key: "{capability.key()}",
                            div { class: "stack",
                                span { class: "field-label", "{route_label(capability)}" }
                                span { class: "field-hint mono", "{now}" }
                            }
                            div { class: "row",
                                Select { value: provider, choices,
                                    onchange: move |to| draft.edit(|c| route_to(c, capability, to)) }
                                input { class: "dx-input mono", placeholder: "the provider's own model",
                                    value: "{model}",
                                    oninput: move |e| draft.edit(|c| route_model(c, capability, &e.value())) }
                            }
                        }
                    }
                })}
                {providers.into_iter().map(|(name, provider)| {
                    let (for_url, for_key, gone) = (name.clone(), name.clone(), name.clone());
                    let url_hint = provider.kind.default_url().unwrap_or("https://host (the root, without /v1)");
                    let key_hint = provider.kind.default_key().map_or_else(
                        || "${env:NAME}".to_string(),
                        |name| format!("${{env:{name}}}"),
                    );
                    rsx! {
                        div { class: "field", key: "{name}",
                            div { class: "stack",
                                span { class: "field-label mono", "{name}" }
                                span { class: "field-hint", "{kind_label(provider.kind)}" }
                            }
                            div { class: "row",
                                input { class: "dx-input mono", placeholder: "{url_hint}",
                                    value: "{provider.url.clone().unwrap_or_default()}",
                                    oninput: move |e| draft.edit(|c| if let Some(p) = c.providers.get_mut(&for_url) {
                                        p.url = Some(e.value().trim().to_string()).filter(|u| !u.is_empty());
                                    }) }
                                input { class: "dx-input mono", placeholder: "{key_hint}",
                                    value: "{provider.key.clone().unwrap_or_default()}",
                                    oninput: move |e| draft.edit(|c| if let Some(p) = c.providers.get_mut(&for_key) {
                                        p.key = Some(e.value().trim().to_string()).filter(|k| !k.is_empty());
                                    }) }
                                button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                                    onclick: move |_| draft.edit(|c| remove_provider(c, &gone)),
                                    "Remove" }
                            }
                        }
                    }
                })}
                div { class: "field",
                    span { class: "field-label", "Add a provider" }
                    div { class: "row",
                        button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                            onclick: move |_| draft.edit(|c| add_provider(c, ProviderKind::Jevons)),
                            "A jevons server" }
                        button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                            onclick: move |_| draft.edit(|c| add_provider(c, ProviderKind::Openrouter)),
                            "OpenRouter" }
                        button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                            onclick: move |_| draft.edit(|c| add_provider(c, ProviderKind::OpenaiCompatible)),
                            "An OpenAI-compatible server" }
                    }
                }
                p { class: "muted",
                    "A key written as ${{env:NAME}} is read from the environment. What a provider's decision model takes (extensions, max_questions, min_probability) is set in the settings file."
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
                    div { class: "stack",
                        span { class: "field-label", "Headphones" }
                        span { class: "field-hint", "Windows · a tap still plays and pauses music" }
                    }
                    Switch { checked: d.dictation.headphones,
                        label: "Double-tap Bluetooth headphones to start live dictation; it ends at a pause".to_string(),
                        onchange: move |on| draft.edit(|c| c.dictation.headphones = on) }
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
                    input { class: "dx-input narrow mono", value: "{tokens}",
                        oninput: move |e| draft.set_typed(Field::Tokens, e.value()) }
                }
            }
        }

        div { class: "dx-card", "data-card": "voice",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Voice" }
                    div { class: "dx-card-description", "Answers read aloud. Starting a take stops the voice, and so do Esc, the bubble's Stop and the tray's Cancel" }
                }
            }
            div { class: "dx-card-content",
                div { class: "field",
                    span { class: "field-label", "Speak answers" }
                    Switch { checked: d.voice.speak,
                        label: "Read aloud the answers of the flows that ask for it (off: they show in the bubble)".to_string(),
                        onchange: move |on| draft.edit(|c| c.voice.speak = on) }
                }
                div { class: "field",
                    span { class: "field-label", "Output device" }
                    Select { value: d.voice.output.clone(), choices: outputs,
                        onchange: move |o| draft.edit(|c| c.voice.output = o) }
                }
                div { class: "field",
                    div { class: "stack",
                        span { class: "field-label", "Voice" }
                        span { class: "field-hint", "As the voice route's provider names it" }
                    }
                    if let Some(choices) = voice_choices {
                        Select { value: voice_chosen, choices,
                            onchange: move |v: Option<String>| draft.edit(|c| route_voice(c, v.as_deref().unwrap_or_default())) }
                    } else {
                        input { class: "dx-input mono", placeholder: "the provider's own voice",
                            value: "{voice_name}",
                            oninput: move |e| draft.edit(|c| route_voice(c, &e.value())) }
                    }
                }
                if !voice_served {
                    p { class: "muted", "Nothing serves the voice yet, so answers show in the bubble: download a voice model in the Models tab, or route Voice to a provider above." }
                }
                if let Some(provider) = &voice_elsewhere {
                    p { class: "warn", "The voice route goes to {provider}: the text of each answer read aloud is sent there." }
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

        div { class: "dx-card", "data-soon": if d.automation.enabled { "false" } else { "true" },
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Automations" }
                    div { class: "dx-card-description", "Record a task once, then run it again from the tray, a hotkey or by saying it" }
                }
                if !d.automation.enabled {
                    {badge("Coming soon", "secondary")}
                }
            }
            if !d.automation.enabled {
                div { class: "dx-card-content",
                    p { class: "muted",
                        "Automations are off for now: jevons records and runs none. To try them, set "
                        span { class: "mono", "enabled = true" }
                        " under "
                        span { class: "mono", "[automation]" }
                        " in jevons-desktop.toml and start jevons again."
                    }
                }
            } else {
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
                    input { class: "dx-input narrow mono", value: "{chars}",
                        oninput: move |e| draft.set_typed(Field::Chars, e.value()) }
                }
                div { class: "field",
                    span { class: "field-label", "Clipboard" }
                    Switch { checked: d.privacy.read_clipboard, label: "Include the clipboard in the context".to_string(),
                        onchange: move |on| draft.edit(|c| c.privacy.read_clipboard = on) }
                }
                div { class: "field",
                    span { class: "field-label", "API log" }
                    Switch { checked: d.log_api,
                        label: "Write every decision and generation request and response to logs/api.log".to_string(),
                        onchange: move |on| draft.edit(|c| c.log_api = on) }
                }
                if d.log_api {
                    p { class: "warn", "The log holds what you say and the text of your screen, in full. Turn it off when you are done debugging." }
                }
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_edited_in_the_panel_leave_what_is_default_out_of_the_file() {
        let mut config = DesktopConfig::default();
        add_provider(&mut config, ProviderKind::Openrouter);
        add_provider(&mut config, ProviderKind::Openrouter);
        add_provider(&mut config, ProviderKind::Jevons);
        let names: Vec<&String> = config.providers.keys().collect();
        assert_eq!(names, ["jevons", "openrouter", "openrouter-2"]);
        // Decisions go to OpenRouter, which needs a model named.
        route_to(&mut config, Capability::Decision, Some("openrouter".into()));
        assert!(config.check_routes().is_err());
        route_model(&mut config, Capability::Decision, " typesafe/jev-1.13 ");
        assert_eq!(config.check_routes(), Ok(()));
        let decision = config.routes.decision.clone().unwrap();
        assert_eq!(decision.model.as_deref(), Some("typesafe/jev-1.13"));
        // Another provider is asked for its own model, not the last one's.
        route_to(&mut config, Capability::Decision, Some("jevons".into()));
        assert_eq!(config.routes.decision.clone().unwrap().model, None);
        // Back on the app's models, the route is left out; so is Realtime following speech.
        route_to(&mut config, Capability::Decision, Some(EMBEDDED.into()));
        route_to(&mut config, Capability::Realtime, None);
        assert_eq!(config.routes, Default::default());
        // A model named for the embedded provider is a route of its own.
        route_model(&mut config, Capability::Generation, "gemma");
        assert_eq!(
            config.routes.generation,
            Some(RouteTo {
                provider: EMBEDDED.into(),
                model: Some("gemma".into()),
                voice: None,
            })
        );
        route_model(&mut config, Capability::Generation, "");
        assert_eq!(config.routes.generation, None);
        // Removing a provider sends what it served back to the default.
        route_to(&mut config, Capability::Speech, Some("jevons".into()));
        remove_provider(&mut config, "jevons");
        assert_eq!(config.routes.speech, None);
        assert!(!config.providers.contains_key("jevons"));
        assert_eq!(config.check_routes(), Ok(()));
    }

    #[test]
    fn the_voice_to_speak_in_is_the_voice_route_s_and_goes_with_its_provider() {
        let mut config = DesktopConfig::default();
        // Named for the app's own model, it is a route of its own.
        route_voice(&mut config, " lector ");
        assert_eq!(
            config.routes.voice,
            Some(RouteTo {
                provider: EMBEDDED.into(),
                model: None,
                voice: Some("lector".into()),
            })
        );
        assert_eq!(config.check_routes(), Ok(()));
        // Cleared, the route is what a route left out means, and is not written.
        route_voice(&mut config, "");
        assert_eq!(config.routes, Default::default());
        // A voice is a provider's: another provider is asked for its own until one is typed.
        add_provider(&mut config, ProviderKind::Jevons);
        route_voice(&mut config, "lector");
        route_to(&mut config, Capability::Voice, Some("jevons".into()));
        assert_eq!(config.routes.voice.clone().unwrap().voice, None);
        route_voice(&mut config, "reader");
        route_model(&mut config, Capability::Voice, "qwen3-tts-latest");
        let voice = config.routes.voice.clone().unwrap();
        assert_eq!(
            (
                voice.provider.as_str(),
                voice.model.as_deref(),
                voice.voice.as_deref()
            ),
            ("jevons", Some("qwen3-tts-latest"), Some("reader"))
        );
        assert_eq!(config.check_routes(), Ok(()));
    }

    #[test]
    fn the_voices_a_provider_lists_are_chosen_among_and_others_are_typed() {
        // A provider that does not say which voices it has: the name is typed.
        assert_eq!(voice_choices(None, None), None);
        assert_eq!(voice_choices(Some(&Voices::default()), Some("nova")), None);
        // The app's own model lists them: its own voice first, by name, then each one.
        let listed = Voices {
            names: vec!["lector".into(), "reader".into()],
            default: "lector".into(),
        };
        let labels = |named: Option<&str>| -> Vec<(Option<String>, String)> {
            voice_choices(Some(&listed), named)
                .unwrap()
                .into_iter()
                .map(|choice| (choice.value, choice.label))
                .collect()
        };
        let own = (None, "The model's own (lector)".to_string());
        let each = [
            (Some("lector".to_string()), "lector".to_string()),
            (Some("reader".to_string()), "reader".to_string()),
        ];
        assert_eq!(
            labels(None),
            [own.clone(), each[0].clone(), each[1].clone()]
        );
        assert_eq!(labels(Some("reader")).len(), 3);
        // A voice the settings name that the model does not have stays the choice, and says so.
        assert_eq!(
            labels(Some("lucia"))[3],
            (Some("lucia".to_string()), "lucia (not listed)".to_string())
        );
    }

    #[test]
    fn saving_applies_only_the_edits_onto_settings_changed_meanwhile() {
        let base = DesktopConfig::default();
        // Edited here: the hotkey, and the live hotkey cleared.
        let mut edited = base.clone();
        edited.dictation.hotkey = "F8".into();
        edited.dictation.live_hotkey = None;
        edited.voice.speak = false;
        edited.voice.output = Some("Headphones".into());
        // Meanwhile, elsewhere: the tray's live feedback, an approval, a branch hotkey.
        let mut latest = base.clone();
        latest.dictation.live_feedback = !base.dictation.live_feedback;
        latest
            .automation
            .approved
            .insert("slack-post".into(), "abc".into());
        latest
            .dictation
            .branch_hotkeys
            .insert("ask".into(), "F7".into());
        let merged = merge(&base, &edited, &latest);
        assert_eq!(merged.dictation.hotkey, "F8");
        assert_eq!(merged.dictation.live_hotkey, None);
        assert!(!merged.voice.speak);
        assert_eq!(merged.voice.output.as_deref(), Some("Headphones"));
        assert_eq!(
            merged.dictation.live_feedback,
            latest.dictation.live_feedback
        );
        assert_eq!(merged.automation.approved["slack-post"], "abc");
        assert_eq!(merged.dictation.branch_hotkeys["ask"], "F7");
    }
}
