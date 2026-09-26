//! Runtime, dictation and privacy settings, applied and saved together.

use crate::agent::{Command, View};
use eframe::egui;
use jevons_desktop_core::config::{DesktopConfig, Mode};
use jevons_desktop_core::context::Privacy;
use tokio::sync::mpsc::UnboundedSender;

pub struct State {
    draft: DesktopConfig,
    bind: String,
    show_key: bool,
}

impl State {
    pub fn new(config: DesktopConfig) -> Self {
        Self {
            bind: config.server.bind.to_string(),
            draft: config,
            show_key: false,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, view: &View, commands: &UnboundedSender<Command>) {
        let draft = &mut self.draft;
        ui.heading("Runtime");
        if let Some(status) = &view.runtime {
            ui.horizontal(|ui| {
                ui.label(status.describe());
                if let crate::runtime::Status::Ready {
                    base_url,
                    exposed: true,
                } = status
                    && ui.small_button("Copy base URL").clicked()
                {
                    ui.ctx().copy_text(format!("{base_url}/v1"));
                }
            });
        }
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut draft.server.mode,
                Mode::Embedded,
                "Run the models in this app",
            );
            ui.radio_value(&mut draft.server.mode, Mode::Remote, "Use a jevons server");
        });
        egui::Grid::new("server")
            .num_columns(2)
            .show(ui, |ui| match draft.server.mode {
                Mode::Embedded => {
                    ui.label("Expose the API");
                    ui.checkbox(
                        &mut draft.server.expose,
                        "Serve other clients (OpenAI SDK, scripts)",
                    );
                    ui.end_row();
                    ui.add_enabled_ui(draft.server.expose, |ui| ui.label("Address"));
                    ui.add_enabled_ui(draft.server.expose, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.bind).desired_width(120.0));
                            ui.label("port");
                            ui.add(egui::DragValue::new(&mut draft.server.port).range(1..=65535));
                        });
                    });
                    ui.end_row();
                    ui.add_enabled_ui(draft.server.expose, |ui| ui.label("API key"));
                    ui.add_enabled_ui(draft.server.expose, |ui| {
                        ui.horizontal(|ui| {
                            let key = draft.server.api_key.get_or_insert_default();
                            ui.add(egui::TextEdit::singleline(key).password(!self.show_key));
                            ui.checkbox(&mut self.show_key, "show");
                        });
                    });
                    ui.end_row();
                    if draft.server.expose && draft.exposed_key().is_none() {
                        ui.label("");
                        ui.colored_label(
                            ui.visuals().warn_fg_color,
                            "Without a key, anyone who can reach the port can use the API.",
                        );
                        ui.end_row();
                    }
                }
                Mode::Remote => {
                    ui.label("Server URL");
                    ui.text_edit_singleline(&mut draft.server.remote_url);
                    ui.end_row();
                    ui.label("API key");
                    let key = draft.server.remote_key.get_or_insert_default();
                    ui.add(egui::TextEdit::singleline(key).password(true));
                    ui.end_row();
                }
            });

        ui.separator();
        ui.heading("Dictation");
        egui::Grid::new("dictation").num_columns(2).show(ui, |ui| {
            let dictation = &mut draft.dictation;
            ui.label("Dictation hotkey")
                .on_hover_text("Tap to toggle dictation, hold while speaking");
            super::hotkey::field(ui, "hotkey", &mut dictation.hotkey, false);
            ui.end_row();
            ui.label("Inspector hotkey");
            let mut inspector = dictation.inspector_hotkey.clone().unwrap_or_default();
            if super::hotkey::field(ui, "inspector-hotkey", &mut inspector, true) {
                dictation.inspector_hotkey = Some(inspector).filter(|h| !h.is_empty());
            }
            ui.end_row();
            ui.label("Microphone");
            egui::ComboBox::from_id_salt("microphone")
                .selected_text(dictation.microphone.as_deref().unwrap_or("Default"))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut dictation.microphone, None, "Default");
                    for device in &view.devices {
                        ui.selectable_value(
                            &mut dictation.microphone,
                            Some(device.name.clone()),
                            &device.name,
                        );
                    }
                });
            ui.end_row();
            ui.label("Language");
            let mut language = dictation.language.clone().unwrap_or_default();
            if ui
                .add(egui::TextEdit::singleline(&mut language).hint_text("detect"))
                .changed()
            {
                dictation.language = Some(language.trim().to_lowercase()).filter(|l| !l.is_empty());
            }
            ui.end_row();
            ui.label("Decide");
            ui.checkbox(
                &mut dictation.decide,
                "Ask the decision model which action to take",
            );
            ui.end_row();
            ui.label("Rewrite threshold");
            ui.add(egui::Slider::new(
                &mut dictation.generation_threshold,
                0.0..=1.0,
            ))
            .on_hover_text(
                "Below this probability that the text needs editing, it is typed as heard",
            );
            ui.end_row();
            ui.label("Max output tokens");
            ui.add(egui::DragValue::new(&mut dictation.max_output_tokens).range(16..=8192));
            ui.end_row();
        });

        ui.collapsing("Dictate with a profile", |ui| {
            ui.label("A hotkey per profile starts a take with that profile, whatever the context matches.");
            egui::Grid::new("profile-hotkeys").num_columns(2).show(ui, |ui| {
                for profile in view.profiles.iter() {
                    let id = &profile.spec.id;
                    ui.label(profile.display_name());
                    let mut hotkey = draft.dictation.profile_hotkeys.get(id).cloned().unwrap_or_default();
                    if super::hotkey::field(ui, ("profile-hotkey", id), &mut hotkey, true) {
                        if hotkey.is_empty() {
                            draft.dictation.profile_hotkeys.remove(id);
                        } else {
                            draft.dictation.profile_hotkeys.insert(id.clone(), hotkey);
                        }
                    }
                    ui.end_row();
                }
            });
        });

        ui.separator();
        ui.heading("Privacy");
        egui::Grid::new("privacy").num_columns(2).show(ui, |ui| {
            let privacy: &mut Privacy = &mut draft.privacy;
            ui.label("Context characters");
            ui.add(egui::DragValue::new(&mut privacy.max_context_chars).range(0..=20000))
                .on_hover_text("The most text kept from each field of the focused element");
            ui.end_row();
            ui.label("Clipboard");
            ui.checkbox(
                &mut privacy.read_clipboard,
                "Include the clipboard in the context",
            );
            ui.end_row();
        });

        ui.separator();
        let bind = self.bind.trim().parse();
        if bind.is_err() {
            ui.colored_label(
                ui.visuals().error_fg_color,
                "The address must be an IP address",
            );
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(bind.is_ok(), egui::Button::new("Apply and save"))
                .clicked()
            {
                let mut config = self.draft.clone();
                config.server.bind = bind.expect("checked above");
                // Keep the model selections, which the Models tab edits.
                config.models = view.config.models.clone();
                config.server.api_key = config.server.api_key.filter(|k| !k.is_empty());
                config.server.remote_key = config.server.remote_key.filter(|k| !k.is_empty());
                let _ = commands.send(Command::Apply(Box::new(config)));
            }
            if ui.button("Revert").clicked() {
                *self = Self::new(view.config.clone());
            }
            ui.label(view.config_file.display().to_string());
        });
    }
}
