//! Where inference runs: the models loaded in this process (served over HTTP on loopback, and
//! to other clients when exposed), or a remote jevons server.
//!
//! The runtime thread owns a Tokio runtime and the loaded models. Applying new settings only
//! rebinds the listener when the models did not change, so exposing the API or changing its
//! port keeps the models in memory.

use jevons_desktop_core::client::Client;
use jevons_desktop_core::config::{DesktopConfig, Mode, ModelRef, Models};
use jevons_desktop_core::pipeline::ServiceModels;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// No models are selected.
    NoModels,
    Loading,
    /// The embedded API is serving on `base_url`; `exposed` when other clients may use it.
    #[cfg_attr(not(feature = "embedded"), allow(dead_code))]
    Ready {
        base_url: String,
        exposed: bool,
    },
    /// Using a remote server.
    Remote {
        url: String,
    },
    Failed(String),
}

impl Status {
    pub fn describe(&self) -> String {
        match self {
            Self::NoModels => "No models selected: choose them in Settings".into(),
            Self::Loading => "Loading models…".into(),
            Self::Ready {
                base_url,
                exposed: true,
            } => format!("Serving the API on {base_url}"),
            Self::Ready { .. } => "Models loaded (API private to this app)".into(),
            Self::Remote { url } => format!("Using {url}"),
            Self::Failed(e) => format!("Failed: {e}"),
        }
    }
}

/// What the pipeline needs to reach the runtime.
#[derive(Clone, Debug)]
pub struct Connection {
    pub client: Client,
    pub models: ServiceModels,
    pub realtime: bool,
}

struct Shared {
    status: Status,
    connection: Option<Connection>,
}

/// Handle to the runtime thread.
#[derive(Clone)]
pub struct Runtime {
    shared: Arc<Mutex<Shared>>,
    apply: mpsc::Sender<Option<(DesktopConfig, PathBuf)>>,
}

impl Runtime {
    /// Starts the runtime thread; `changed` runs after every status change.
    pub fn start(changed: impl Fn() + Send + 'static) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            status: Status::NoModels,
            connection: None,
        }));
        let (apply, requests) = mpsc::channel();
        let state = shared.clone();
        std::thread::Builder::new()
            .name("runtime".into())
            .spawn(move || run(requests, state, changed))
            .expect("the runtime thread starts");
        Self { shared, apply }
    }

    /// Applies `config` (read from `file`, which relative model paths resolve against).
    pub fn apply(&self, config: &DesktopConfig, file: &Path) {
        let _ = self.apply.send(Some((config.clone(), file.to_path_buf())));
    }

    /// Unloads the models and ends the thread.
    pub fn shutdown(&self) {
        let _ = self.apply.send(None);
    }

    pub fn status(&self) -> Status {
        self.shared.lock().expect("the runtime lock").status.clone()
    }

    pub fn connection(&self) -> Option<Connection> {
        self.shared
            .lock()
            .expect("the runtime lock")
            .connection
            .clone()
    }
}

fn run(
    requests: mpsc::Receiver<Option<(DesktopConfig, PathBuf)>>,
    shared: Arc<Mutex<Shared>>,
    changed: impl Fn(),
) {
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("runtime-io")
        .build()
        .expect("the Tokio runtime builds");
    let set = |status: Status, connection: Option<Connection>| {
        let status = match status {
            Status::Failed(e) => Status::Failed(explain(e)),
            other => other,
        };
        match &status {
            Status::Failed(_) => tracing::error!(status = %status.describe(), "Runtime"),
            _ => tracing::info!(status = %status.describe(), "Runtime"),
        }
        let mut shared = shared.lock().expect("the runtime lock");
        shared.status = status;
        shared.connection = connection;
        drop(shared);
        changed();
    };
    let mut embedded = embedded::Embedded::default();
    while let Ok(Some((config, file))) = requests.recv() {
        // Only the latest settings matter.
        let (config, file) = std::iter::from_fn(|| requests.try_recv().ok())
            .map_while(|r| r)
            .last()
            .unwrap_or((config, file));
        match config.server.mode {
            Mode::Remote => {
                tokio.block_on(embedded.unload());
                let key = config
                    .server
                    .remote_key
                    .clone()
                    .or_else(|| std::env::var("TYPESAFE_API_KEY").ok());
                let client = Client::new(&config.server.remote_url, key);
                let url = client.base().to_string();
                match tokio.block_on(client.health()) {
                    Ok(health) => {
                        let models = ServiceModels {
                            speech: health.services.speech,
                            decision: health.services.decision,
                            generative: health.services.generative,
                        };
                        let connection = Connection {
                            client,
                            models,
                            realtime: true,
                        };
                        set(Status::Remote { url }, Some(connection));
                    }
                    Err(e) => set(Status::Failed(e.to_string()), None),
                }
            }
            Mode::Embedded => {
                let settings = match runtime_settings(&config.models, &file) {
                    Ok(Some(text)) => text,
                    Ok(None) => {
                        tokio.block_on(embedded.unload());
                        set(Status::NoModels, None);
                        continue;
                    }
                    Err(e) => {
                        set(Status::Failed(e), None);
                        continue;
                    }
                };
                if embedded.needs_load(&settings) {
                    set(Status::Loading, None);
                }
                match embedded.apply(&tokio, &settings, &file, &config) {
                    Ok((status, connection)) => set(status, Some(connection)),
                    Err(e) => set(Status::Failed(e), None),
                }
            }
        }
    }
    tokio.block_on(embedded.unload());
}

/// Adds what to do about failures the user can fix outside the app.
fn explain(error: String) -> String {
    let gpu = [
        "hipMemGetInfo",
        "hipInit",
        "Could not load HIP",
        "DriverError",
    ];
    if gpu.iter().any(|marker| error.contains(marker)) {
        format!(
            "The GPU is not usable through HIP ({error}). Check that the AMD driver matches the \
             HIP SDK: its hipInfo tool must run without errors."
        )
    } else {
        error
    }
}

/// The jevons-rs settings for the selected models, or `None` when none is selected.
pub fn runtime_settings(models: &Models, file: &Path) -> Result<Option<String>, String> {
    if let Some(path) = &models.runtime_config {
        let path = if path.is_relative() {
            file.parent().unwrap_or(Path::new(".")).join(path)
        } else {
            path.clone()
        };
        return std::fs::read_to_string(&path)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display()));
    }
    let mut table = toml::Table::new();
    let mut model_table = toml::Table::new();
    let mut services = toml::Table::new();
    let mut names: Vec<(ModelRef, String)> = Vec::new();
    let mut name_for = |model: &ModelRef| -> String {
        if let Some((_, name)) = names
            .iter()
            .find(|(m, _)| m.path == model.path && m.mmproj == model.mmproj)
        {
            return name.clone();
        }
        let name = format!("model{}", names.len());
        let mut entry = toml::Table::new();
        entry.insert("path".into(), model.path.display().to_string().into());
        if let Some(mmproj) = &model.mmproj {
            entry.insert("mmproj".into(), mmproj.display().to_string().into());
        }
        model_table.insert(name.clone(), entry.into());
        names.push((model.clone(), name.clone()));
        name
    };
    for (service, model) in [
        ("generative", &models.generative),
        ("decision", &models.decision),
        ("speech", &models.speech),
    ] {
        if let Some(model) = model {
            let mut entry = toml::Table::new();
            entry.insert("model".into(), name_for(model).into());
            if service == "speech" {
                entry.insert("realtime".into(), models.realtime.into());
            }
            services.insert(service.into(), entry.into());
        }
    }
    if services.is_empty() {
        return Ok(None);
    }
    table.insert("models".into(), model_table.into());
    table.insert("services".into(), services.into());
    Ok(Some(toml::to_string(&table).expect("settings serialize")))
}

#[cfg(feature = "embedded")]
mod embedded {
    use super::{Connection, Status};
    use jevons_api::config::Settings;
    use jevons_api::{AppState, Workers};
    use jevons_desktop_core::client::Client;
    use jevons_desktop_core::config::DesktopConfig;
    use jevons_desktop_core::pipeline::ServiceModels;
    use std::net::{IpAddr, SocketAddr};
    use std::path::Path;
    use std::sync::Arc;
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;

    struct Loaded {
        settings: String,
        state: AppState,
        workers: Workers,
    }

    struct Listener {
        address: SocketAddr,
        exposed: bool,
        key: Option<String>,
        stop: oneshot::Sender<()>,
        task: JoinHandle<std::io::Result<()>>,
    }

    #[derive(Default)]
    pub struct Embedded {
        loaded: Option<Loaded>,
        listener: Option<Listener>,
    }

    impl Embedded {
        pub fn needs_load(&self, settings: &str) -> bool {
            self.loaded.as_ref().is_none_or(|l| l.settings != settings)
        }

        async fn stop_listener(&mut self) {
            if let Some(listener) = self.listener.take() {
                let _ = listener.stop.send(());
                let _ = listener.task.await;
            }
        }

        pub async fn unload(&mut self) {
            self.stop_listener().await;
            if let Some(loaded) = self.loaded.take() {
                drop(loaded.state);
                if let Err(e) = loaded.workers.join().await {
                    tracing::warn!(error = %e, "A model worker did not stop cleanly");
                }
            }
        }

        pub fn apply(
            &mut self,
            tokio: &tokio::runtime::Runtime,
            settings: &str,
            file: &Path,
            config: &DesktopConfig,
        ) -> Result<(Status, Connection), String> {
            if self.needs_load(settings) {
                tokio.block_on(self.unload());
                let parsed = Settings::parse(settings, file).map_err(|e| e.to_string())?;
                let (state, workers) = tokio
                    .block_on(jevons_api::load(&parsed))
                    .map_err(|e| e.to_string())?;
                self.loaded = Some(Loaded {
                    settings: settings.to_string(),
                    state,
                    workers,
                });
            }
            let server = &config.server;
            let exposed_address = SocketAddr::new(server.bind, server.port);
            let exposed_key = config.exposed_key();
            let keep = self.listener.as_ref().is_some_and(|l| {
                if server.expose {
                    l.exposed && l.address == exposed_address && l.key == exposed_key
                } else {
                    // A private listener keeps its port and key while it runs.
                    !l.exposed
                }
            });
            if !keep {
                tokio.block_on(self.stop_listener());
                let (address, key) = if server.expose {
                    // Without a key the exposed API is open, as jevons-rs is without one.
                    (exposed_address, exposed_key)
                } else {
                    let key = format!(
                        "{}{}",
                        uuid::Uuid::new_v4().simple(),
                        uuid::Uuid::new_v4().simple()
                    );
                    (SocketAddr::from(([127, 0, 0, 1], 0)), Some(key))
                };
                let loaded = self.loaded.as_ref().expect("the models are loaded");
                let listener = tokio
                    .block_on(tokio::net::TcpListener::bind(address))
                    .map_err(|e| format!("Cannot listen on {address}: {e}"))?;
                let address = listener.local_addr().map_err(|e| e.to_string())?;
                let state = AppState {
                    api_key: key.as_deref().map(Arc::from),
                    ..loaded.state.clone()
                };
                let (stop, stopped) = oneshot::channel::<()>();
                let task = tokio.spawn(jevons_api::serve(listener, state, async {
                    let _ = stopped.await;
                }));
                tracing::info!(%address, exposed = server.expose, "Serving the embedded API");
                self.listener = Some(Listener {
                    address,
                    exposed: server.expose,
                    key,
                    stop,
                    task,
                });
            }
            Ok(self.connection())
        }

        fn connection(&self) -> (Status, Connection) {
            let listener = self.listener.as_ref().expect("the listener runs");
            let state = &self.loaded.as_ref().expect("the models are loaded").state;
            let host = match listener.address.ip() {
                IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::from([127, 0, 0, 1]),
                IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::from([127, 0, 0, 1]),
                ip => ip,
            };
            let base_url = format!("http://{}", SocketAddr::new(host, listener.address.port()));
            let connection = Connection {
                client: Client::new(&base_url, listener.key.clone()),
                models: ServiceModels {
                    speech: state.speech.as_ref().map(|s| s.model_id.clone()),
                    decision: state.decision.as_ref().map(|s| s.model_id.clone()),
                    generative: state.generative.as_ref().map(|s| s.model_id.clone()),
                },
                realtime: state.speech.as_ref().is_some_and(|s| s.realtime),
            };
            let exposed_url = format!("http://{}", listener.address);
            let status = Status::Ready {
                base_url: if listener.exposed {
                    exposed_url
                } else {
                    base_url
                },
                exposed: listener.exposed,
            };
            (status, connection)
        }
    }
}

#[cfg(not(feature = "embedded"))]
mod embedded {
    use super::{Connection, Status};
    use jevons_desktop_core::config::DesktopConfig;
    use std::path::Path;

    #[derive(Default)]
    pub struct Embedded {}

    impl Embedded {
        pub fn needs_load(&self, _: &str) -> bool {
            false
        }

        pub async fn unload(&mut self) {}

        pub fn apply(
            &mut self,
            _: &tokio::runtime::Runtime,
            _: &str,
            _: &Path,
            _: &DesktopConfig,
        ) -> Result<(Status, Connection), String> {
            Err("this build has no embedded runtime; use a remote server".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(path: &str) -> Option<ModelRef> {
        Some(ModelRef {
            path: path.into(),
            mmproj: None,
            catalog: None,
        })
    }

    #[test]
    fn services_on_the_same_model_share_one_model_entry() {
        let models = Models {
            generative: model("/m/nemotron"),
            decision: model("/m/nemotron"),
            speech: model("/m/parakeet"),
            realtime: false,
            ..Models::default()
        };
        let text = runtime_settings(&models, Path::new("/c/jevons-desktop.toml"))
            .unwrap()
            .unwrap();
        let table: toml::Table = toml::from_str(&text).unwrap();
        assert_eq!(table["models"].as_table().unwrap().len(), 2);
        assert_eq!(
            table["services"]["generative"]["model"],
            table["services"]["decision"]["model"]
        );
        assert_eq!(
            table["services"]["speech"]["realtime"].as_bool(),
            Some(false)
        );
    }

    #[test]
    fn hip_failures_say_how_to_check_the_driver() {
        let message = explain("DriverError { op: \"hipMemGetInfo\", status: 719 }".into());
        assert!(message.contains("hipInfo"), "{message}");
        assert_eq!(explain("models.x: not found".into()), "models.x: not found");
    }

    #[test]
    fn no_selected_model_means_no_runtime() {
        assert_eq!(
            runtime_settings(&Models::default(), Path::new("x.toml")).unwrap(),
            None
        );
    }

    #[cfg(feature = "embedded")]
    #[test]
    fn generated_settings_are_valid_for_the_runtime() {
        let models = Models {
            decision: model("/m/nemotron"),
            speech: model("/m/parakeet"),
            ..Models::default()
        };
        let file = Path::new("/c/jevons-desktop.toml");
        let text = runtime_settings(&models, file).unwrap().unwrap();
        let settings = jevons_api::config::Settings::parse(&text, file).unwrap();
        assert!(settings.services.speech.unwrap().realtime);
    }
}
