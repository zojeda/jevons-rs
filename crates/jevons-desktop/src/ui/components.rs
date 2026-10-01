//! Components in the Dioxus Components design, written for Blitz: no JavaScript, no popover API,
//! and no mounted or pointer events (dioxus-native 0.7), so menus and dialogs are in-window
//! overlays and focus comes from clicks.

use dioxus::prelude::*;
use serde_json::Value;

/// An icon, drawn as SVG: Blitz has no fallback font for symbol glyphs such as ▾ or ✔.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    ChevronDown,
    ChevronUp,
    ChevronRight,
    Check,
    Cross,
    Plus,
}

pub fn icon(icon: Icon) -> Element {
    let (path, color) = match icon {
        Icon::ChevronDown => ("M6 9l6 6 6-6", "#a1a1a1"),
        Icon::ChevronUp => ("M6 15l6-6 6 6", "#a1a1a1"),
        Icon::ChevronRight => ("M9 6l6 6-6 6", "#a1a1a1"),
        Icon::Check => ("M5 12.5l4.5 4.5L19 7.5", "#b6fae3"),
        Icon::Cross => ("M7 7l10 10M17 7L7 17", "#ffb4b4"),
        Icon::Plus => ("M12 5v14M5 12h14", "#22e6f2"),
    };
    rsx! {
        svg {
            class: "icon",
            width: "14",
            height: "14",
            view_box: "0 0 24 24",
            fill: "none",
            stroke: color,
            stroke_width: "2.5",
            stroke_linecap: "round",
            stroke_linejoin: "round",
            path { d: path }
        }
    }
}

/// A labelled on/off switch.
#[component]
pub fn Switch(checked: bool, label: String, onchange: EventHandler<bool>) -> Element {
    rsx! {
        div { class: "switch-row", onclick: move |_| onchange.call(!checked),
            button {
                class: "dx-switch",
                "data-state": if checked { "checked" } else { "unchecked" },
                span { class: "dx-switch-thumb" }
            }
            span { "{label}" }
        }
    }
}

/// One entry of a [`Select`].
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub value: Option<String>,
    pub label: String,
}

/// A select drawn as a button and a listbox below it.
#[component]
pub fn Select(
    value: Option<String>,
    choices: Vec<Choice>,
    onchange: EventHandler<Option<String>>,
) -> Element {
    let mut open = use_signal(|| false);
    let current = choices
        .iter()
        .find(|c| c.value == value)
        .map_or_else(|| "—".to_string(), |c| c.label.clone());
    rsx! {
        div { class: "select",
            button {
                class: "dx-button select-trigger",
                "data-style": "outline",
                "data-size": "sm",
                onclick: move |_| open.set(!open()),
                span { "{current}" }
                {icon(if open() { Icon::ChevronUp } else { Icon::ChevronDown })}
            }
            if open() {
                div { class: "select-list",
                    {choices.iter().map(|choice| {
                        let picked = choice.value.clone();
                        let selected = (choice.value == value).to_string();
                        rsx! {
                            button {
                                class: "select-option",
                                "data-selected": selected,
                                onclick: move |_| {
                                    open.set(false);
                                    onchange.call(picked.clone());
                                },
                                "{choice.label}"
                            }
                        }
                    })}
                }
            }
        }
    }
}

/// A hotkey: click the field, then press the combination (Escape cancels).
#[component]
pub fn HotkeyField(value: String, optional: bool, onchange: EventHandler<String>) -> Element {
    let mut recording = use_signal(|| false);
    let shown = if recording() {
        "Press the keys… (Esc cancels)".to_string()
    } else if value.is_empty() {
        "Not set".to_string()
    } else {
        value.clone()
    };
    rsx! {
        div { class: "row",
            input {
                class: "dx-input mono",
                readonly: true,
                value: "{shown}",
                "data-recording": recording().to_string(),
                onclick: move |_| recording.set(true),
                onblur: move |_| recording.set(false),
                onkeydown: move |event: KeyboardEvent| {
                    if !recording() {
                        return;
                    }
                    event.prevent_default();
                    if event.key() == Key::Escape {
                        recording.set(false);
                    } else if let Some(accelerator) = accelerator(event.code(), event.modifiers()) {
                        recording.set(false);
                        onchange.call(accelerator);
                    }
                },
            }
            if optional && !value.is_empty() {
                button {
                    class: "dx-button",
                    "data-style": "ghost",
                    "data-size": "sm",
                    onclick: move |_| onchange.call(String::new()),
                    "Clear"
                }
            }
        }
    }
}

/// The accelerator for a key press, as global-hotkey parses it. A plain key needs a modifier,
/// except function keys; modifier keys alone are ignored.
pub fn accelerator(code: Code, modifiers: Modifiers) -> Option<String> {
    let name = code.to_string();
    if ["Control", "Alt", "Shift", "Meta", "Super", "OS"]
        .iter()
        .any(|m| name.starts_with(m))
    {
        return None;
    }
    let function_key = name.len() > 1 && name.starts_with('F') && name[1..].parse::<u8>().is_ok();
    let mut parts = Vec::new();
    if modifiers.contains(Modifiers::CONTROL) {
        parts.push("Ctrl");
    }
    if modifiers.contains(Modifiers::ALT) {
        parts.push("Alt");
    }
    if modifiers.contains(Modifiers::SHIFT) {
        parts.push("Shift");
    }
    if modifiers.contains(Modifiers::META) {
        parts.push("Super");
    }
    if parts.is_empty() && !function_key {
        return None;
    }
    let key = name
        .strip_prefix("Key")
        .or_else(|| name.strip_prefix("Digit"))
        .unwrap_or(&name)
        .to_string();
    Some(
        parts
            .into_iter()
            .map(String::from)
            .chain([key])
            .collect::<Vec<_>>()
            .join("+"),
    )
}

/// A section that opens and closes; `open` is its initial state, and `status` adds a check or a
/// cross before the title.
#[component]
pub fn Collapsible(
    title: String,
    subtitle: Option<String>,
    open: bool,
    #[props(default)] status: Option<bool>,
    children: Element,
) -> Element {
    let mut expanded = use_signal(|| open);
    rsx! {
        div { class: "dx-accordion-item",
            button {
                class: "dx-accordion-trigger",
                onclick: move |_| expanded.set(!expanded()),
                {icon(if expanded() { Icon::ChevronDown } else { Icon::ChevronRight })}
                if let Some(passed) = status {
                    {icon(if passed { Icon::Check } else { Icon::Cross })}
                }
                span { class: "grow", "{title}" }
                if let Some(subtitle) = &subtitle {
                    span { class: "muted", "{subtitle}" }
                }
            }
            if expanded() {
                div { class: "dx-accordion-content", {children} }
            }
        }
    }
}

/// A confirmation dialog over the window.
#[component]
pub fn Confirm(
    title: String,
    message: String,
    confirm: String,
    onconfirm: EventHandler<()>,
    oncancel: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "dx-dialog-overlay", onclick: move |_| oncancel.call(()),
            div { class: "dx-dialog", onclick: move |event| event.stop_propagation(),
                h2 { "{title}" }
                p { class: "muted", "{message}" }
                div { class: "dx-dialog-actions",
                    button {
                        class: "dx-button",
                        "data-style": "outline",
                        onclick: move |_| oncancel.call(()),
                        "Cancel"
                    }
                    button {
                        class: "dx-button",
                        "data-style": "destructive",
                        onclick: move |_| onconfirm.call(()),
                        "{confirm}"
                    }
                }
            }
        }
    }
}

/// A small label.
pub fn badge(text: &str, style: &'static str) -> Element {
    rsx! {
        span { class: "dx-badge", "data-style": style, "{text}" }
    }
}

/// A progress bar, `fraction` from 0 to 1.
pub fn progress(fraction: f32) -> Element {
    let percent = (fraction.clamp(0.0, 1.0) * 100.0).round();
    rsx! {
        div { class: "dx-progress",
            div { class: "dx-progress-indicator", style: "width: {percent}%" }
        }
    }
}

/// A JSON value as a tree that opens node by node.
#[component]
pub fn JsonTree(value: Value) -> Element {
    let rows: Vec<(String, Value)> = match &value {
        Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        other => vec![("value".into(), other.clone())],
    };
    rsx! {
        div { class: "tree",
            {rows.into_iter().map(|(label, value)| rsx! { JsonNode { label, value, depth: 0 } })}
        }
    }
}

#[component]
fn JsonNode(label: String, value: Value, depth: usize) -> Element {
    let mut open = use_signal(|| depth < 1);
    let children: Option<Vec<(String, Value)>> = match &value {
        Value::Object(map) if !map.is_empty() => {
            Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        }
        Value::Array(items) if !items.is_empty() => Some(
            items
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect(),
        ),
        _ => None,
    };
    match children {
        Some(children) => {
            let count = match &value {
                Value::Array(items) => format!(" [{}]", items.len()),
                _ => String::new(),
            };
            rsx! {
                div {
                    button { class: "tree-toggle", onclick: move |_| open.set(!open()),
                        {icon(if open() { Icon::ChevronDown } else { Icon::ChevronRight })}
                        span { class: "tree-key", "{label}" }
                        "{count}"
                    }
                    if open() {
                        div { class: "tree-children",
                            {children.into_iter().map(|(label, value)| rsx! {
                                JsonNode { label, value, depth: depth + 1 }
                            })}
                        }
                    }
                }
            }
        }
        None => {
            let text = match &value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            rsx! {
                div { class: "tree-row",
                    span { class: "tree-key", "{label}:" }
                    span { class: "tree-value", "{text}" }
                }
            }
        }
    }
}

/// Copies `text` to the clipboard; the error says why it could not.
pub fn copy(text: &str) -> Result<(), String> {
    arboard::Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text))
        .map_err(|e| e.to_string())
}

/// A button that copies `text` and then says whether it did, until the text changes.
#[component]
pub fn CopyButton(text: String, label: String, #[props(default)] disabled: bool) -> Element {
    let mut done = use_signal(|| None::<(String, bool)>);
    let shown = match &*done.read() {
        Some((copied, true)) if *copied == text => "Copied".to_string(),
        Some((copied, false)) if *copied == text => "Copy failed".to_string(),
        _ => label.clone(),
    };
    rsx! {
        button { class: "dx-button", "data-style": "outline", "data-size": "sm", disabled,
            onclick: move |_| done.set(Some((text.clone(), copy(&text).is_ok()))),
            "{shown}"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use global_hotkey::hotkey::HotKey;

    #[test]
    fn recorded_combinations_parse_as_global_hotkeys() {
        let ctrl_alt = Modifiers::CONTROL | Modifiers::ALT;
        for code in [Code::Space, Code::KeyA, Code::Digit1, Code::F9] {
            let text = accelerator(code, ctrl_alt).unwrap();
            assert!(text.parse::<HotKey>().is_ok(), "{text}");
        }
        assert_eq!(
            accelerator(Code::KeyL, ctrl_alt).as_deref(),
            Some("Ctrl+Alt+L")
        );
        assert_eq!(
            accelerator(Code::F9, Modifiers::empty()).as_deref(),
            Some("F9")
        );
        assert_eq!(accelerator(Code::KeyA, Modifiers::empty()), None);
        assert_eq!(accelerator(Code::ControlLeft, ctrl_alt), None);
    }
}
