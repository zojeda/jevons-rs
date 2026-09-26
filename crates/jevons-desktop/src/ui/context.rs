//! The live context and the profile resolution for it.

use crate::agent::{Command, View};
use eframe::egui;
use jevons_desktop_core::profile::{Check, Resolution};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Default)]
pub struct State {
    raw: bool,
}

pub fn show(
    ui: &mut egui::Ui,
    view: &View,
    frozen: &mut bool,
    state: &mut State,
    commands: &UnboundedSender<Command>,
) {
    ui.horizontal(|ui| {
        ui.checkbox(frozen, "Freeze");
        if ui
            .button("Capture in 3 s")
            .on_hover_text("Switch to the application you want to inspect")
            .clicked()
        {
            *frozen = true;
            let _ = commands.send(Command::CaptureContextIn(Duration::from_secs(3)));
        }
        ui.checkbox(&mut state.raw, "Raw JSON");
        ui.separator();
        ui.label(format!(
            "context: {} · delivery: {}",
            view.context_backend, view.sink_backend
        ));
    });
    if view.context_paused {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "Context capture is paused (tray menu).",
        );
    }
    if let Some(error) = &view.context_error {
        ui.colored_label(ui.visuals().error_fg_color, error);
    }
    let Some(context) = &view.context else {
        ui.label("Switch to another application: the context of the focused window appears here.");
        return;
    };
    ui.separator();
    let value = serde_json::to_value(context).unwrap_or_default();
    if state.raw {
        let mut text = serde_json::to_string_pretty(&value).unwrap_or_default();
        ui.add(
            egui::TextEdit::multiline(&mut text)
                .code_editor()
                .desired_width(f32::INFINITY),
        );
    } else {
        egui::Grid::new("context")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let row = |ui: &mut egui::Ui, key: &str, value: &str| {
                    ui.strong(key);
                    ui.label(value);
                    ui.end_row();
                };
                row(ui, "Application", &context.app.process_name);
                row(ui, "Window", &context.window.title);
                if let Some(url) = &context.url {
                    row(ui, "Address", url);
                }
                if let Some(e) = &context.focused {
                    row(ui, "Role", &e.role);
                    row(ui, "Name", &e.name);
                    if let Some(id) = &e.automation_id {
                        row(ui, "Automation id", id);
                    }
                    row(ui, "Editable", if e.is_editable { "yes" } else { "no" });
                    if e.is_password {
                        row(ui, "Password", "yes (text never read)");
                    }
                    for (key, value) in [
                        ("Selection", &e.selection),
                        ("Before caret", &e.before_caret),
                        ("After caret", &e.after_caret),
                        ("Value", &e.value_excerpt),
                    ] {
                        if let Some(value) = value {
                            row(ui, key, &excerpt(value));
                        }
                    }
                }
                for (key, value) in &context.extras {
                    row(ui, key, &excerpt(value));
                }
            });
        for error in &context.errors {
            ui.colored_label(ui.visuals().warn_fg_color, error);
        }
    }
    if let Some(resolution) = &view.resolution {
        ui.separator();
        resolution_table(ui, resolution);
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
pub fn resolution_table(ui: &mut egui::Ui, resolution: &Resolution) {
    ui.horizontal(|ui| {
        ui.heading("Profile");
        ui.strong(&resolution.profile);
        if let Some(destination) = &resolution.destination {
            ui.label("→");
            ui.strong(destination);
        }
        if resolution.forced {
            ui.label("(forced from the tray)");
        }
        if !resolution.tied.is_empty() {
            ui.label(format!(
                "tied with {}: the decision model chooses",
                resolution.tied.join(", ")
            ));
        }
    });
    for profile in &resolution.trace {
        let title = format!(
            "{} {}  priority {} · {} rules",
            if profile.matched { "✔" } else { "✘" },
            profile.id,
            if profile.priority == i32::MIN {
                "lowest".to_string()
            } else {
                profile.priority.to_string()
            },
            profile.specificity
        );
        egui::CollapsingHeader::new(title)
            .id_salt(("profile", &profile.id))
            .default_open(profile.id == resolution.profile)
            .show(ui, |ui| {
                checks(ui, &profile.checks);
                for destination in &profile.destinations {
                    ui.label(format!(
                        "{} destination {}  priority {}",
                        if destination.matched { "✔" } else { "✘" },
                        destination.id,
                        destination.priority
                    ));
                    ui.indent(("destination", &profile.id, &destination.id), |ui| {
                        checks(ui, &destination.checks)
                    });
                }
            });
    }
}

fn checks(ui: &mut egui::Ui, checks: &[Check]) {
    if checks.is_empty() {
        ui.label("No rules: matches everything.");
        return;
    }
    egui::Grid::new(ui.next_auto_id())
        .striped(true)
        .show(ui, |ui| {
            for check in checks {
                ui.label(if check.passed { "✔" } else { "✘" });
                ui.monospace(check.rule);
                ui.monospace(&check.pattern);
                ui.label(check.value.as_deref().unwrap_or("(nothing)"));
                ui.end_row();
            }
        });
}
