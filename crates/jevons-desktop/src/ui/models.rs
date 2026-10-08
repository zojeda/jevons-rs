//! The models folder, the catalog with optional downloads, and the model each service uses.

use super::Ctx;
use super::components::{Collapsible, Confirm, Switch, badge, progress};
use crate::agent::{Command, open_folder};
use crate::config::{DesktopConfig, ModelRef};
use dioxus::prelude::*;
use jevons_desktop_core::catalog::{self, CatalogEntry, Service};
use jevons_desktop_core::download::{Hub, Progress};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

type Downloads = HashMap<String, Arc<Mutex<Download>>>;

/// The downloads of this run, outside the page: leaving the Models tab unmounts the page while
/// the downloads go on, and coming back must still show their progress.
static DOWNLOADS: std::sync::LazyLock<Mutex<Downloads>> =
    std::sync::LazyLock::new(|| Mutex::new(Downloads::new()));

fn download_state(id: &str) -> Option<Arc<Mutex<Download>>> {
    DOWNLOADS
        .lock()
        .expect("the downloads lock")
        .get(id)
        .cloned()
}

fn catalog_file(config_file: &Path) -> PathBuf {
    config_file
        .parent()
        .unwrap_or(Path::new("."))
        .join("models.toml")
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

/// Downloads `entry` on its own thread, waking the window as it progresses. When it finishes, the
/// runtime reloads, so the new model is used without a restart.
fn start(ctx: Ctx, entry: CatalogEntry, folder: PathBuf) {
    let download = Arc::new(Mutex::new(Download::default()));
    DOWNLOADS
        .lock()
        .expect("the downloads lock")
        .insert(entry.id.clone(), download.clone());
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
                    let mut woken = Instant::now();
                    let progress = |p: Progress| {
                        download.lock().expect("the download lock").progress = p;
                        if woken.elapsed() > Duration::from_millis(250) {
                            woken = Instant::now();
                            super::wake();
                        }
                    };
                    hub.download(&entry, &files, &folder, cancel, progress)
                        .await
                        .map_err(|e| e.to_string())
                })
            });
        let succeeded = result.is_ok();
        {
            let mut download = download.lock().expect("the download lock");
            download.finished = true;
            download.error = result.err();
        }
        if succeeded {
            ctx.send(Command::ReloadRuntime);
        }
        super::wake();
    });
}

#[component]
pub fn ModelsPage(rev: u64) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let config_file = ctx.view.lock().expect("the view lock").config_file.clone();
    let mut entries = use_signal(|| catalog::load(&catalog_file(&config_file)));
    let mut confirm_delete = use_signal(|| None::<CatalogEntry>);
    let mut repo = use_signal(String::new);
    let mut revision = use_signal(|| "main".to_string());
    let mut globs = use_signal(|| "*.json, *.safetensors".to_string());
    let mut model_file = use_signal(String::new);
    let mut speech = use_signal(|| false);

    let config = ctx.view.lock().expect("the view lock").config.clone();
    let folder = config.models_folder();
    let (catalog, catalog_error) = entries();
    let apply = move |ctx: &Ctx, config: DesktopConfig| ctx.send(Command::Apply(Box::new(config)));
    let memory: f32 = catalog
        .iter()
        .filter(|e| {
            SERVICES.iter().any(|(s, _)| {
                selection(&config, *s).and_then(|m| m.catalog.as_deref()) == Some(e.id.as_str())
            })
        })
        .map(|e| e.memory_gb)
        .sum();

    let choose_ctx = ctx.clone();
    let choose_config = config.clone();
    let open = folder.clone();
    let realtime_ctx = ctx.clone();
    let realtime_config = config.clone();

    rsx! {
        div { class: "spread",
            div { class: "stack",
                h2 { "Models" }
                span { class: "muted mono", "{folder.display()}" }
            }
            div { class: "row",
                button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                    onclick: move |_| {
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            let mut config = choose_config.clone();
                            config.models.folder = Some(dir);
                            apply(&choose_ctx, config);
                        }
                    },
                    "Choose folder…"
                }
                button { class: "dx-button", "data-style": "ghost", "data-size": "sm",
                    onclick: move |_| {
                        let _ = std::fs::create_dir_all(&open);
                        open_folder(&open);
                    },
                    "Open"
                }
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Selected models" }
                    div { class: "dx-card-description",
                        "A service without a selection uses the first downloaded catalog model that serves it: DiffusionGemma, then Parakeet for speech."
                    }
                }
                if memory > 0.0 {
                    {badge(&format!("≈ {memory:.0} GB while loaded"), "secondary")}
                }
            }
            div { class: "dx-card-content",
                if let Some(path) = &config.models.runtime_config {
                    p { class: "warn", "Loading {path.display()} instead; these selections are ignored." }
                }
                {SERVICES.iter().map(|&(service, name)| {
                    let current = selection(&config, service)
                        .map_or_else(|| "automatic".to_string(), |m| m.path.display().to_string());
                    let is_set = selection(&config, service).is_some();
                    let pick_ctx = ctx.clone();
                    let pick_config = config.clone();
                    let (file_ctx, file_config) = (ctx.clone(), config.clone());
                    let clear_ctx = ctx.clone();
                    let clear_config = config.clone();
                    rsx! {
                        div { class: "field", key: "{name}",
                            span { class: "field-label", "{name}" }
                            div { class: "row",
                                span { class: "mono grow wrap-anywhere", "{current}" }
                                // A GGUF file (DiffusionGemma), or a checkpoint folder (Nemotron,
                                // Parakeet): one dialog each, so cancelling one opens nothing else.
                                if service != Service::Speech {
                                    button { class: "dx-button", "data-style": "outline", "data-size": "xs",
                                        onclick: move |_| {
                                            let picked = rfd::FileDialog::new().add_filter("GGUF", &["gguf"]).pick_file();
                                            if let Some(path) = picked {
                                                let mut config = file_config.clone();
                                                *selection_mut(&mut config, service) =
                                                    Some(ModelRef { path, mmproj: None, catalog: None });
                                                apply(&file_ctx, config);
                                            }
                                        },
                                        "GGUF file…"
                                    }
                                }
                                button { class: "dx-button", "data-style": "outline", "data-size": "xs",
                                    onclick: move |_| {
                                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                            let mut config = pick_config.clone();
                                            *selection_mut(&mut config, service) =
                                                Some(ModelRef { path, mmproj: None, catalog: None });
                                            apply(&pick_ctx, config);
                                        }
                                    },
                                    "Checkpoint folder…"
                                }
                                if is_set {
                                    button { class: "dx-button", "data-style": "ghost", "data-size": "xs",
                                        onclick: move |_| {
                                            let mut config = clear_config.clone();
                                            *selection_mut(&mut config, service) = None;
                                            apply(&clear_ctx, config);
                                        },
                                        "Automatic"
                                    }
                                }
                            }
                        }
                    }
                })}
                Switch { checked: config.models.realtime, label: "Realtime transcription (live text while speaking)".to_string(),
                    onchange: move |on| {
                        let mut config = realtime_config.clone();
                        config.models.realtime = on;
                        apply(&realtime_ctx, config);
                    } }
            }
        }

        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Catalog" }
                    div { class: "dx-card-description", "Downloads from Hugging Face resume if interrupted and are checked against SHA-256" }
                }
            }
            div { class: "dx-card-content",
                if let Some(error) = catalog_error.clone() {
                    p { class: "error-text", "{error}" }
                }
                {catalog.iter().map(|entry| {
                    let services: Vec<&str> = SERVICES.iter().filter(|(s, _)| entry.serves(*s)).map(|(_, n)| *n).collect();
                    let services = services.join(", ");
                    let state = download_state(&entry.id);
                    // Also downloading when the runtime fetches it as a default model at startup.
                    let running = state.as_ref().is_some_and(|d| !d.lock().expect("the download lock").finished)
                        || jevons_desktop_core::download::is_running(entry, &folder);
                    let (fraction, file, error) = state.as_ref().map_or((0.0, String::new(), None), |d| {
                        let d = d.lock().expect("the download lock");
                        let fraction = if d.progress.total > 0 { d.progress.done as f32 / d.progress.total as f32 } else { 0.0 };
                        (fraction, d.progress.file.clone(), d.error.clone())
                    });
                    let ready = entry.is_ready(&folder);
                    let partial = !ready && entry.dir(&folder).exists();
                    let percent = (fraction * 100.0).round();
                    let download_entry = entry.clone();
                    let download_ctx = ctx.clone();
                    let download_folder = folder.clone();
                    let cancel = state.clone();
                    let delete_entry = entry.clone();
                    let use_config = config.clone();
                    let use_ctx = ctx.clone();
                    rsx! {
                        div { class: "dx-card", key: "{entry.id}",
                            div { class: "dx-card-header",
                                div {
                                    div { class: "dx-card-title", "{entry.name}" }
                                    div { class: "dx-card-description", "{entry.repo} · {entry.license} · ≈ {entry.memory_gb} GB · {services}" }
                                }
                                if ready { {badge("ready", "success")} }
                                else if running { {badge("downloading", "accent")} }
                                else if partial { {badge("partial", "warning")} }
                                else { {badge("not downloaded", "outline")} }
                            }
                            div { class: "dx-card-content",
                                if running {
                                    {progress(fraction)}
                                    div { class: "spread",
                                        span { class: "muted mono", "{file} · {percent}%" }
                                        button { class: "dx-button", "data-style": "ghost", "data-size": "xs",
                                            onclick: move |_| if let Some(d) = &cancel {
                                                d.lock().expect("the download lock").cancel.store(true, Ordering::Relaxed);
                                            },
                                            "Cancel"
                                        }
                                    }
                                }
                                if let Some(error) = error {
                                    p { class: "error-text", "{error}" }
                                }
                                div { class: "row",
                                    if !ready && !running {
                                        button { class: "dx-button", "data-style": "accent", "data-size": "sm",
                                            onclick: move |_| start(download_ctx.clone(), download_entry.clone(), download_folder.clone()),
                                            if partial { "Resume download" } else { "Download" }
                                        }
                                    }
                                    if ready {
                                        {SERVICES.iter().filter(|(s, _)| entry.serves(*s)).map(|&(service, name)| {
                                            let chosen = selection(&use_config, service).and_then(|m| m.catalog.as_deref()) == Some(entry.id.as_str());
                                            let entry = entry.clone();
                                            let folder = folder.clone();
                                            let config = use_config.clone();
                                            let ctx = use_ctx.clone();
                                            rsx! {
                                                button { class: "dx-button", "data-size": "sm", key: "{name}",
                                                    "data-style": if chosen { "secondary" } else { "outline" },
                                                    disabled: chosen,
                                                    onclick: move |_| {
                                                        let mut config = config.clone();
                                                        *selection_mut(&mut config, service) = Some(ModelRef {
                                                            path: entry.model_path(&folder),
                                                            mmproj: entry.mmproj_path(&folder),
                                                            catalog: Some(entry.id.clone()),
                                                        });
                                                        apply(&ctx, config);
                                                    },
                                                    if chosen { "Used for {name}" } else { "Use for {name}" }
                                                }
                                            }
                                        })}
                                    }
                                    if (ready || partial) && !running {
                                        button { class: "dx-button", "data-style": "ghost", "data-size": "sm",
                                            onclick: move |_| confirm_delete.set(Some(delete_entry.clone())),
                                            "Delete"
                                        }
                                    }
                                }
                            }
                        }
                    }
                })}
            }
        }

        div { class: "dx-accordion",
            Collapsible { title: "Add a Hugging Face model".to_string(), subtitle: None, open: false,
                div { class: "field",
                    span { class: "field-label", "Repository" }
                    input { class: "dx-input mono", placeholder: "owner/name", value: "{repo}", oninput: move |e| repo.set(e.value()) }
                }
                div { class: "field",
                    span { class: "field-label", "Revision" }
                    input { class: "dx-input mono narrow", value: "{revision}", oninput: move |e| revision.set(e.value()) }
                }
                div { class: "field",
                    span { class: "field-label", "Files" }
                    input { class: "dx-input mono", value: "{globs}", oninput: move |e| globs.set(e.value()) }
                }
                div { class: "field",
                    span { class: "field-label", "Model file" }
                    input { class: "dx-input mono", placeholder: "GGUF only", value: "{model_file}", oninput: move |e| model_file.set(e.value()) }
                }
                div { class: "field",
                    span { class: "field-label", "Speech model" }
                    Switch { checked: speech(), label: "Transcribes audio (otherwise a language model)".to_string(), onchange: move |on| speech.set(on) }
                }
                div { class: "row",
                    button { class: "dx-button", "data-style": "secondary", "data-size": "sm", disabled: !repo().contains('/'),
                        onclick: move |_| {
                            let repo = repo();
                            let entry = CatalogEntry {
                                id: repo.rsplit('/').next().unwrap_or_default().to_lowercase(),
                                name: repo.clone(),
                                services: if speech() { vec![Service::Speech] } else { vec![Service::Generative, Service::Decision] },
                                repo,
                                revision: revision(),
                                files: globs().split(',').map(|g| g.trim().to_string()).filter(|g| !g.is_empty()).collect(),
                                model_file: Some(model_file()).filter(|f| !f.is_empty()),
                                mmproj_file: None,
                                license: String::new(),
                                memory_gb: 0.0,
                                extra: Vec::new(),
                            };
                            let file = catalog_file(&config_file);
                            match add_to_catalog(&file, &entry) {
                                Ok(()) => entries.set(catalog::load(&file)),
                                Err(e) => entries.write().1 = Some(e),
                            }
                        },
                        "Add to catalog"
                    }
                }
            }
        }

        if let Some(entry) = confirm_delete() {
            {
                let dir = entry.dir(&folder);
                let message = format!("This removes {} and every file in it.", dir.display());
                rsx! {
                    Confirm {
                        title: format!("Delete {}?", entry.name),
                        message,
                        confirm: "Delete".to_string(),
                        onconfirm: move |_| {
                            let _ = std::fs::remove_dir_all(&dir);
                            confirm_delete.set(None);
                        },
                        oncancel: move |_| confirm_delete.set(None),
                    }
                }
            }
        }
    }
}

/// Appends `entry` to the user's `models.toml`.
fn add_to_catalog(file: &Path, entry: &CatalogEntry) -> Result<(), String> {
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
