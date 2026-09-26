//! The inspector window: the live context and why a profile matches, the recent takes, the
//! profiles, and the settings (runtime, models, dictation, privacy). Closing it hides it; the
//! tray keeps running.

mod context;
mod models;
mod profiles;
mod settings;
mod takes;

use crate::agent::{Command, SharedView};
use eframe::egui;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Context,
    Takes,
    Profiles,
    Settings,
    Models,
}

pub struct App {
    view: SharedView,
    commands: UnboundedSender<Command>,
    tab: Tab,
    frozen: bool,
    last_refresh: Instant,
    context: context::State,
    profiles: profiles::State,
    settings: settings::State,
    models: models::State,
}

impl App {
    pub fn new(view: SharedView, commands: UnboundedSender<Command>) -> Self {
        let (config, config_file) = {
            let view = view.lock().expect("the view lock");
            (view.config.clone(), view.config_file.clone())
        };
        Self {
            view,
            commands,
            tab: Tab::Context,
            frozen: false,
            last_refresh: Instant::now() - Duration::from_secs(1),
            context: context::State::default(),
            profiles: profiles::State::default(),
            settings: settings::State::new(config),
            models: models::State::new(&config_file),
        }
    }

    fn send(&self, command: Command) {
        let _ = self.commands.send(command);
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let (quit, show) = {
            let mut view = self.view.lock().expect("the view lock");
            (view.quit, std::mem::take(&mut view.show_window))
        };
        if quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if show {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.tab == Tab::Context
            && !self.frozen
            && self.last_refresh.elapsed() >= Duration::from_millis(500)
        {
            self.last_refresh = Instant::now();
            self.send(Command::RefreshContext);
        }
        ui.ctx().request_repaint_after(Duration::from_millis(500));

        egui::Panel::top("tabs").show(ui, |ui| {
            ui.horizontal(|ui| {
                for (tab, label) in [
                    (Tab::Context, "Context"),
                    (Tab::Takes, "Takes"),
                    (Tab::Profiles, "Profiles"),
                    (Tab::Settings, "Settings"),
                    (Tab::Models, "Models"),
                ] {
                    ui.selectable_value(&mut self.tab, tab, label);
                }
            });
        });
        egui::Panel::bottom("status").show(ui, |ui| self.status(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Context => {
                    let view = self.view.lock().expect("the view lock");
                    let commands = &self.commands;
                    context::show(ui, &view, &mut self.frozen, &mut self.context, commands);
                }
                Tab::Takes => {
                    let view = self.view.lock().expect("the view lock");
                    takes::show(ui, &view);
                }
                Tab::Profiles => {
                    let view = self.view.lock().expect("the view lock");
                    profiles::show(ui, &view, &mut self.profiles, &self.commands);
                }
                Tab::Settings => {
                    let view = self.view.lock().expect("the view lock");
                    self.settings.show(ui, &view, &self.commands);
                }
                Tab::Models => {
                    let config = self.view.lock().expect("the view lock").config.clone();
                    self.models.show(ui, &config, &self.commands);
                }
            });
        });
    }
}

impl App {
    fn status(&self, ui: &mut egui::Ui) {
        let view = self.view.lock().expect("the view lock");
        ui.horizontal_wrapped(|ui| {
            ui.label(view.tray.tooltip());
            ui.separator();
            if let Some(status) = &view.runtime {
                ui.label(status.describe());
                ui.separator();
            }
            ui.label(format!("hotkey {}", view.config.dictation.hotkey));
            if let Some(error) = &view.hotkey_error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
            if let Some(notice) = &view.notice {
                ui.separator();
                ui.colored_label(ui.visuals().warn_fg_color, notice);
            }
        });
        if view.dictating || !view.live_transcript.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("Hearing:");
                ui.label(&view.live_transcript);
            });
        }
        if !view.live_output.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.strong("Writing:");
                ui.label(&view.live_output);
            });
        }
    }
}

/// A JSON value as a collapsible tree.
pub(crate) fn json_tree(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &serde_json::Value,
) {
    use serde_json::Value;
    fn node(ui: &mut egui::Ui, key: &str, value: &Value, path: String) {
        match value {
            Value::Object(map) if !map.is_empty() => {
                egui::CollapsingHeader::new(key)
                    .id_salt(&path)
                    .default_open(path.matches('/').count() < 2)
                    .show(ui, |ui| {
                        for (k, v) in map {
                            node(ui, k, v, format!("{path}/{k}"));
                        }
                    });
            }
            Value::Array(items) if !items.is_empty() => {
                egui::CollapsingHeader::new(format!("{key} [{}]", items.len()))
                    .id_salt(&path)
                    .show(ui, |ui| {
                        for (i, v) in items.iter().enumerate() {
                            node(ui, &i.to_string(), v, format!("{path}/{i}"));
                        }
                    });
            }
            Value::String(s) => {
                ui.horizontal_wrapped(|ui| {
                    ui.monospace(format!("{key}:"));
                    ui.label(s);
                });
            }
            other => {
                ui.monospace(format!("{key}: {other}"));
            }
        }
    }
    ui.push_id(id, |ui| {
        if let Value::Object(map) = value {
            for (k, v) in map {
                node(ui, k, v, k.clone());
            }
        } else {
            node(ui, "value", value, "value".into());
        }
    });
}
