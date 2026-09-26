//! A hotkey field: click it, then press the combination.

use eframe::egui;

/// Shows `value` as a button; clicked, it records the next key combination (Escape cancels).
/// Returns true when the value changed.
pub fn field(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &mut String,
    optional: bool,
) -> bool {
    let id = ui.make_persistent_id(id);
    let recording = ui.data(|d| d.get_temp::<bool>(id)).unwrap_or(false);
    let mut changed = false;
    ui.horizontal(|ui| {
        let label = if recording {
            "Press keys… (Esc cancels)".to_string()
        } else if value.is_empty() {
            "Not set".to_string()
        } else {
            value.clone()
        };
        let button =
            ui.add(egui::Button::new(egui::RichText::new(label).monospace()).selected(recording));
        if button.clicked() {
            ui.data_mut(|d| d.insert_temp(id, !recording));
        }
        if recording {
            let pressed = ui.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => Some((*key, *modifiers)),
                    _ => None,
                })
            });
            if let Some((key, modifiers)) = pressed {
                if key == egui::Key::Escape {
                    ui.data_mut(|d| d.insert_temp(id, false));
                } else if let Some(accelerator) = accelerator(key, modifiers) {
                    *value = accelerator;
                    changed = true;
                    ui.data_mut(|d| d.insert_temp(id, false));
                }
            }
            ui.ctx().request_repaint();
        }
        if optional && !value.is_empty() && ui.small_button("✖").on_hover_text("Clear").clicked()
        {
            value.clear();
            changed = true;
        }
    });
    changed
}

/// The accelerator for `key` with `modifiers`, as global-hotkey parses it. A plain key needs a
/// modifier, except function keys.
fn accelerator(key: egui::Key, modifiers: egui::Modifiers) -> Option<String> {
    let name = key.name();
    let function_key = name.len() > 1 && name.starts_with('F') && name[1..].parse::<u8>().is_ok();
    if !(modifiers.ctrl || modifiers.alt || modifiers.shift || modifiers.mac_cmd || function_key) {
        return None;
    }
    let mut parts = Vec::new();
    if modifiers.ctrl {
        parts.push("Ctrl");
    }
    if modifiers.alt {
        parts.push("Alt");
    }
    if modifiers.shift {
        parts.push("Shift");
    }
    if modifiers.mac_cmd {
        parts.push("Super");
    }
    parts.push(name);
    Some(parts.join("+"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use global_hotkey::hotkey::HotKey;

    #[test]
    fn recorded_combinations_parse_as_global_hotkeys() {
        let ctrl_alt = egui::Modifiers {
            ctrl: true,
            alt: true,
            ..egui::Modifiers::NONE
        };
        for key in [
            egui::Key::Space,
            egui::Key::A,
            egui::Key::Num1,
            egui::Key::F9,
        ] {
            let text = accelerator(key, ctrl_alt).unwrap();
            assert!(text.parse::<HotKey>().is_ok(), "{text}");
        }
        assert_eq!(
            accelerator(egui::Key::F9, egui::Modifiers::NONE).as_deref(),
            Some("F9")
        );
        assert_eq!(accelerator(egui::Key::A, egui::Modifiers::NONE), None);
    }
}
