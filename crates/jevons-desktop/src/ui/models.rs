//! The models folder, the catalog with optional downloads, and the model each service uses.

use crate::agent::{Command, open_folder};
use eframe::egui;
use jevons_desktop_core::catalog::{self, CatalogEntry, Service};
use jevons_desktop_core::config::{DesktopConfig, ModelRef};
use jevons_desktop_core::download::{Hub, Progress};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;

const SERVICES: [(Service, &str); 3] = [
    (Service::Generative, "generative"),
    (Service::Decision, "decision"),
    (Service::Speech, "speech"),
];

#[derive(Default)]
struct Download {
    progress: Progress,
    error: Option<String>,
    finished: bool,
    cancel: Arc<AtomicBool>,
}

pub struct State {
    entries: Vec<CatalogEntry>,
    catalog_error: Option<String>,
    downloads: HashMap<String, Arc<Mutex<Download>>>,
    custom: CatalogEntry,
    custom_files: String,
    confirm_delete: Option<String>,
    catalog_file: PathBuf,
}

fn catalog_file(config_file: &std::path::Path) -> PathBuf {
    config_file
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("models.toml")
}

impl State {
    pub fn new(config_file: &std::path::Path) -> Self {
        let file = catalog_file(config_file);
        let (entries, catalog_error) = catalog::load(&file);
        Self {
            entries,
            catalog_error,
            downloads: HashMap::new(),
            custom: CatalogEntry {
                id: String::new(),
                name: String::new(),
                services: vec![Service::Generative, Service::Decision],
                repo: String::new(),
                revision: "main".into(),
                files: Vec::new(),
                model_file: None,
                mmproj_file: None,
                license: String::new(),
                memory_gb: 0.0,
            },
            custom_files: "*.json, *.safetensors".into(),
            confirm_delete: None,
            catalog_file: file,
        }
    }

    fn start(&mut self, entry: CatalogEntry, folder: PathBuf) {
        let download = Arc::new(Mutex::new(Download::default()));
        self.downloads.insert(entry.id.clone(), download.clone());
        let cancel = download.lock().expect("the download lock").cancel.clone();
        std::thread::spawn(move || {
            let result = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())
                .and_then(|runtime| {
                    runtime.block_on(async {
                        let hub = Hub::default();
                        let files = hub.plan(&entry).await.map_err(|e| e.to_string())?;
                        let progress = |p: Progress| {
                            download.lock().expect("the download lock").progress = p;
                        };
                        hub.download(&entry, &files, &folder, cancel, progress)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
            let mut download = download.lock().expect("the download lock");
            download.finished = true;
            download.error = result.err();
        });
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        config: &DesktopConfig,
        commands: &UnboundedSender<Command>,
    ) {
        let folder = config.models_folder();
        let apply = |config: DesktopConfig| {
            let _ = commands.send(Command::Apply(Box::new(config)));
        };
        ui.heading("Models folder");
        ui.horizontal(|ui| {
            ui.monospace(folder.display().to_string());
            if ui.button("Choose…").clicked()
                && let Some(dir) = rfd::FileDialog::new().set_directory(&folder).pick_folder()
            {
                let mut config = config.clone();
                config.models.folder = Some(dir);
                apply(config);
            }
            if ui.button("Open").clicked() {
                let _ = std::fs::create_dir_all(&folder);
                open_folder(&folder);
            }
        });

        ui.separator();
        ui.heading("Selected models");
        if let Some(path) = &config.models.runtime_config {
            ui.label(format!(
                "Loading {} instead (see Settings); the selections below are ignored.",
                path.display()
            ));
        }
        egui::Grid::new("selected").num_columns(3).show(ui, |ui| {
            for (service, name) in SERVICES {
                ui.strong(name);
                let selected = selection(config, service);
                ui.label(selected.map_or("none".into(), |m| m.path.display().to_string()));
                ui.horizontal(|ui| {
                    if ui.small_button("Use existing…").clicked() {
                        let pick = if service == Service::Speech {
                            rfd::FileDialog::new().pick_folder()
                        } else {
                            // A GGUF file, or cancel and pick a checkpoint folder.
                            rfd::FileDialog::new()
                                .add_filter("GGUF", &["gguf"])
                                .pick_file()
                                .or_else(|| rfd::FileDialog::new().pick_folder())
                        };
                        if let Some(path) = pick {
                            let mut config = config.clone();
                            *selection_mut(&mut config, service) = Some(ModelRef {
                                path,
                                mmproj: None,
                                catalog: None,
                            });
                            apply(config);
                        }
                    }
                    if selected.is_some() && ui.small_button("Clear").clicked() {
                        let mut config = config.clone();
                        *selection_mut(&mut config, service) = None;
                        apply(config);
                    }
                });
                ui.end_row();
            }
        });
        let mut realtime = config.models.realtime;
        if ui
            .checkbox(
                &mut realtime,
                "Realtime transcription (live text while speaking)",
            )
            .changed()
        {
            let mut config = config.clone();
            config.models.realtime = realtime;
            apply(config);
        }
        let memory: f32 = self
            .entries
            .iter()
            .filter(|e| {
                SERVICES.iter().any(|(s, _)| {
                    selection(config, *s).and_then(|m| m.catalog.as_deref()) == Some(e.id.as_str())
                })
            })
            .map(|e| e.memory_gb)
            .sum();
        if memory > 0.0 {
            ui.label(format!(
                "About {memory:.1} GB while loaded. On an APU this memory is shared with the system."
            ));
        }

        ui.separator();
        ui.heading("Catalog");
        if let Some(error) = &self.catalog_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let mut start = None;
        egui::Grid::new("catalog")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                for entry in &self.entries {
                    ui.vertical(|ui| {
                        ui.strong(&entry.name);
                        ui.small(format!(
                            "{} · {} · ~{} GB",
                            entry.repo, entry.license, entry.memory_gb
                        ));
                    });
                    let services: Vec<&str> = SERVICES
                        .iter()
                        .filter(|(s, _)| entry.serves(*s))
                        .map(|(_, n)| *n)
                        .collect();
                    ui.label(services.join(", "));
                    let download = self
                        .downloads
                        .get(&entry.id)
                        .map(|d| d.lock().expect("the download lock"));
                    let ready = entry.is_ready(&folder);
                    match &download {
                        Some(d) if !d.finished => {
                            let fraction = if d.progress.total > 0 {
                                d.progress.done as f32 / d.progress.total as f32
                            } else {
                                0.0
                            };
                            ui.add(
                                egui::ProgressBar::new(fraction)
                                    .text(format!("{} {:.0}%", d.progress.file, fraction * 100.0))
                                    .desired_width(220.0),
                            );
                            if ui.small_button("Cancel").clicked() {
                                d.cancel.store(true, Ordering::Relaxed);
                            }
                        }
                        _ => {
                            if let Some(error) = download.as_ref().and_then(|d| d.error.clone()) {
                                ui.colored_label(ui.visuals().error_fg_color, error);
                            } else if ready {
                                ui.label("✔ ready");
                            } else if entry.dir(&folder).exists() {
                                ui.label("partial");
                            } else {
                                ui.label("not downloaded");
                            }
                            ui.horizontal(|ui| {
                                if !ready && ui.small_button("Download").clicked() {
                                    start = Some(entry.clone());
                                }
                                if ready {
                                    for (service, name) in SERVICES {
                                        if entry.serves(service)
                                            && ui.small_button(format!("Use for {name}")).clicked()
                                        {
                                            let mut config = config.clone();
                                            *selection_mut(&mut config, service) = Some(ModelRef {
                                                path: entry.model_path(&folder),
                                                mmproj: entry.mmproj_path(&folder),
                                                catalog: Some(entry.id.clone()),
                                            });
                                            apply(config);
                                        }
                                    }
                                }
                                if entry.dir(&folder).exists() {
                                    if self.confirm_delete.as_deref() == Some(&entry.id) {
                                        if ui.small_button("Really delete?").clicked() {
                                            let _ = std::fs::remove_dir_all(entry.dir(&folder));
                                            self.confirm_delete = None;
                                        }
                                    } else if ui.small_button("Delete").clicked() {
                                        self.confirm_delete = Some(entry.id.clone());
                                    }
                                }
                            });
                        }
                    }
                    ui.end_row();
                }
            });
        if let Some(entry) = start {
            self.start(entry, folder.clone());
        }

        ui.separator();
        ui.collapsing("Add a Hugging Face model", |ui| {
            egui::Grid::new("custom").num_columns(2).show(ui, |ui| {
                ui.label("Repository");
                ui.add(egui::TextEdit::singleline(&mut self.custom.repo).hint_text("owner/name"));
                ui.end_row();
                ui.label("Revision");
                ui.text_edit_singleline(&mut self.custom.revision);
                ui.end_row();
                ui.label("Files");
                ui.text_edit_singleline(&mut self.custom_files)
                    .on_hover_text("Comma-separated globs, such as *Q4_K_M.gguf, mmproj-*.gguf");
                ui.end_row();
                ui.label("Model file");
                let mut model_file = self.custom.model_file.clone().unwrap_or_default();
                if ui
                    .add(egui::TextEdit::singleline(&mut model_file).hint_text("GGUF only"))
                    .changed()
                {
                    self.custom.model_file = Some(model_file).filter(|f| !f.is_empty());
                }
                ui.end_row();
                ui.label("Services");
                ui.horizontal(|ui| {
                    let speech = self.custom.services.contains(&Service::Speech);
                    let mut is_speech = speech;
                    ui.radio_value(&mut is_speech, false, "language");
                    ui.radio_value(&mut is_speech, true, "speech");
                    if is_speech != speech {
                        self.custom.services = if is_speech {
                            vec![Service::Speech]
                        } else {
                            vec![Service::Generative, Service::Decision]
                        };
                    }
                });
                ui.end_row();
            });
            let valid = self.custom.repo.contains('/');
            if ui
                .add_enabled(valid, egui::Button::new("Add to catalog"))
                .clicked()
            {
                let mut entry = self.custom.clone();
                entry.id = entry
                    .repo
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_lowercase();
                entry.name = entry.repo.clone();
                entry.files = self
                    .custom_files
                    .split(',')
                    .map(|g| g.trim().to_string())
                    .filter(|g| !g.is_empty())
                    .collect();
                match add_to_catalog(&self.catalog_file, &entry) {
                    Ok(()) => {
                        self.entries.retain(|e| e.id != entry.id);
                        self.entries.push(entry);
                    }
                    Err(e) => self.catalog_error = Some(e),
                }
            }
        });
    }
}

fn selection(config: &DesktopConfig, service: Service) -> Option<&ModelRef> {
    match service {
        Service::Generative => config.models.generative.as_ref(),
        Service::Decision => config.models.decision.as_ref(),
        Service::Speech => config.models.speech.as_ref(),
    }
}

fn selection_mut(config: &mut DesktopConfig, service: Service) -> &mut Option<ModelRef> {
    match service {
        Service::Generative => &mut config.models.generative,
        Service::Decision => &mut config.models.decision,
        Service::Speech => &mut config.models.speech,
    }
}

/// Appends `entry` to the user's `models.toml`.
fn add_to_catalog(file: &std::path::Path, entry: &CatalogEntry) -> Result<(), String> {
    let mut table: toml::Table = std::fs::read_to_string(file)
        .ok()
        .and_then(|t| toml::from_str(&t).ok())
        .unwrap_or_default();
    let models = table
        .entry("models")
        .or_insert_with(|| toml::Value::Array(Vec::new()));
    let toml::Value::Array(models) = models else {
        return Err(format!("{}: models must be an array", file.display()));
    };
    models.retain(|m| m.get("id").and_then(|i| i.as_str()) != Some(&entry.id));
    models.push(toml::Value::try_from(entry).map_err(|e| e.to_string())?);
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        file,
        toml::to_string_pretty(&table).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", file.display()))
}
