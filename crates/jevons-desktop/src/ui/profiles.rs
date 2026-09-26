//! The loaded profiles, their errors, and new profiles drafted from the current context.

use crate::agent::{Command, View, open_folder};
use eframe::egui;
use jevons_desktop_core::profile::draft;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Default)]
pub struct State {
    new_id: String,
    message: Option<String>,
}

pub fn show(
    ui: &mut egui::Ui,
    view: &View,
    state: &mut State,
    commands: &UnboundedSender<Command>,
) {
    let dir = view.config.profiles_dir(&view.config_file);
    ui.horizontal(|ui| {
        ui.label(format!("Folder: {}", dir.display()));
        if ui.button("Open folder").clicked() {
            let _ = std::fs::create_dir_all(&dir);
            open_folder(&dir);
        }
        if ui.button("Reload").clicked() {
            let _ = commands.send(Command::ReloadProfiles);
        }
    });
    for error in &view.profiles.errors {
        ui.colored_label(
            ui.visuals().error_fg_color,
            format!("{}: {}", error.file.display(), error.message),
        );
    }
    ui.separator();
    ui.heading("New profile from the current context");
    match &view.context {
        Some(context) => {
            ui.label(format!(
                "Matches {} · {}",
                context.app.process_name, context.window.title
            ));
            ui.horizontal(|ui| {
                ui.label("id");
                ui.text_edit_singleline(&mut state.new_id);
                let valid = !state.new_id.is_empty()
                    && state
                        .new_id
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
                if ui.add_enabled(valid, egui::Button::new("Create")).clicked() {
                    let file = dir.join(format!("{}.toml", state.new_id));
                    state.message = Some(if file.exists() {
                        format!("{} already exists", file.display())
                    } else {
                        let _ = std::fs::create_dir_all(&dir);
                        match std::fs::write(&file, draft(context, &state.new_id)) {
                            Ok(()) => {
                                let _ = commands.send(Command::ReloadProfiles);
                                open_folder(&file);
                                format!(
                                    "Created {}; edit its instructions and rules",
                                    file.display()
                                )
                            }
                            Err(e) => e.to_string(),
                        }
                    });
                }
            });
            ui.collapsing("Preview", |ui| {
                let mut text = draft(
                    context,
                    if state.new_id.is_empty() {
                        "new"
                    } else {
                        &state.new_id
                    },
                );
                ui.add(
                    egui::TextEdit::multiline(&mut text)
                        .code_editor()
                        .desired_width(f32::INFINITY),
                );
            });
        }
        None => {
            ui.label("Open the Context tab and capture the application first.");
        }
    }
    if let Some(message) = &state.message {
        ui.label(message);
    }
    ui.separator();
    ui.heading("Profiles");
    for profile in view.profiles.iter() {
        let spec = &profile.spec;
        egui::CollapsingHeader::new(format!(
            "{} ({}) · priority {}",
            profile.display_name(),
            spec.id,
            if spec.priority == i32::MIN {
                "lowest".into()
            } else {
                spec.priority.to_string()
            }
        ))
        .id_salt(("profiles", &spec.id))
        .show(ui, |ui| {
            match &profile.source {
                Some(file) => {
                    ui.horizontal(|ui| {
                        ui.label(file.display().to_string());
                        if ui.small_button("Open").clicked() {
                            open_folder(file);
                        }
                    });
                }
                None => {
                    ui.label("Built in: matches everything, at the lowest priority.");
                }
            }
            let text = toml::to_string_pretty(spec).unwrap_or_default();
            ui.monospace(text);
        });
    }
}
