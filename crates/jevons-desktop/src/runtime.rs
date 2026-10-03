//! Where inference runs: each capability goes to the provider its route names. The embedded
//! provider is the models loaded in this process, served over HTTP on loopback to this app
//! alone; the others are servers elsewhere. When the API is exposed, a forwarder serves other
//! clients what the routes serve.
//!
//! The runtime thread owns a Tokio runtime, the loaded models and the forwarder. Exposing the
//! API or changing its port only rebinds the forwarder, so the models stay in memory.

use crate::config::{Capability, DesktopConfig, ModelRef, Models, Provider, ProviderKind};
use jevons_desktop_server::client::{Client, Profile, Route, Routes};
use jevons_desktop_server::forward::{self, Forwarder};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// Nothing serves any capability: no models are selected, and no route goes elsewhere.
    NoModels,
    Loading,
    /// The app's models are loaded. `api` is where other clients reach the API, when it is
    /// exposed.
    Ready {
        api: Option<String>,
    },
    /// Every capability served is on a provider elsewhere.
    Remote {
        providers: Vec<String>,
        api: Option<String>,
    },
    Failed(String),
}

impl Status {
    /// A word or two for the window's top bar; `describe` has the whole story.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NoModels => "No models",
            Self::Loading => "Loading models",
            Self::Ready { api: Some(_) } | Self::Remote { api: Some(_), .. } => "Serving the API",
            Self::Ready { .. } => "Ready",
            Self::Remote { .. } => "Remote providers",
            Self::Failed(_) => "Failed",
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::NoModels => "No models yet: download them in the Models tab".into(),
            Self::Loading => "Loading models…".into(),
            Self::Ready { api: Some(api) } => format!("Serving the API on {api}"),
            Self::Ready { api: None } => "Models loaded (API private to this app)".into(),
            Self::Remote {
                providers,
                api: Some(api),
            } => format!("Using {}; serving the API on {api}", providers.join(", ")),
            Self::Remote { providers, .. } => format!("Using {}", providers.join(", ")),
            Self::Failed(e) => format!("Failed: {e}"),
        }
    }

    /// Where other clients reach the API, while it is exposed.
    pub fn api(&self) -> Option<&str> {
        match self {
            Self::Ready { api } | Self::Remote { api, .. } => api.as_deref(),
            _ => None,
        }
    }
}

/// A provider that answers: how to reach it, and the model it names for each capability
/// (a route that names its own model needs none of them).
#[derive(Clone, Debug)]
pub struct Served {
    pub client: Client,
    pub speech: Option<String>,
    pub decision: Option<String>,
    pub generation: Option<String>,
    /// Whether it streams speech (Realtime).
    pub realtime: bool,
}

impl Served {
    /// A server that does not say what it serves: its routes name their models.
    fn unnamed(client: Client) -> Self {
        Self {
            client,
            speech: None,
            decision: None,
            generation: None,
            realtime: true,
        }
    }

    fn model(&self, capability: Capability) -> Option<String> {
        match capability {
            Capability::Speech => self.speech.clone(),
            Capability::Realtime => self.speech.clone().filter(|_| self.realtime),
            Capability::Decision => self.decision.clone(),
            Capability::Generation => self.generation.clone(),
        }
    }
}

/// Each capability's route over the providers that answer: the route's own model, else the
/// one its provider names. A capability whose provider does not answer, or names no model for
/// it, is not served.
pub fn routes(config: &DesktopConfig, served: &BTreeMap<String, Served>) -> Routes {
    let route = |capability: Capability| {
        let to = config.route(capability)?;
        let provider = config.provider(&to.provider)?;
        let server = served.get(&to.provider)?;
        Some(Route {
            client: server.client.clone(),
            model: to.model.or_else(|| server.model(capability))?,
            provider: to.provider,
            profile: Profile::of(&provider),
        })
    };
    Routes {
        speech: route(Capability::Speech),
        realtime: route(Capability::Realtime),
        decision: route(Capability::Decision),
        generation: route(Capability::Generation),
    }
}

/// The providers the routes use, by name.
fn used(config: &DesktopConfig) -> BTreeMap<String, Provider> {
    Capability::ALL
        .iter()
        .filter_map(|capability| {
            let name = config.route(*capability)?.provider;
            let provider = config.provider(&name)?;
            Some((name, provider))
        })
        .collect()
}

/// The capabilities routed to the models loaded in this app: only their models are loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Local {
    pub speech: bool,
    pub realtime: bool,
    pub decision: bool,
    pub generation: bool,
}

impl Local {
    pub fn of(config: &DesktopConfig) -> Self {
        let embedded = |capability: Capability| {
            config
                .route(capability)
                .and_then(|route| config.provider(&route.provider))
                .is_some_and(|provider| provider.kind == ProviderKind::Embedded)
        };
        Self {
            speech: embedded(Capability::Speech) || embedded(Capability::Realtime),
            realtime: embedded(Capability::Realtime),
            decision: embedded(Capability::Decision),
            generation: embedded(Capability::Generation),
        }
    }

    fn any(self) -> bool {
        self.speech || self.decision || self.generation
    }
}

/// The API other clients use, while exposed: the forwarder on `bind:port`.
struct Exposed {
    address: SocketAddr,
    key: Option<String>,
    forwarder: Forwarder,
    stop: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

/// Serves `routes` to other clients when the settings expose the API, and stops serving when
/// they do not. A listener with the same address and key is kept, with the routes as they
/// are now. Returns where the API is.
fn expose(
    tokio: &tokio::runtime::Runtime,
    exposed: &mut Option<Exposed>,
    config: &DesktopConfig,
    routes: Option<&Routes>,
) -> Result<Option<String>, String> {
    let address = SocketAddr::new(config.server.bind, config.server.port);
    let key = config.exposed_key();
    let routes = routes.filter(|_| config.server.expose);
    let keep = exposed
        .as_ref()
        .is_some_and(|e| routes.is_some() && e.address == address && e.key == key);
    if !keep && let Some(old) = exposed.take() {
        let _ = old.stop.send(());
        let _ = tokio.block_on(old.task);
    }
    let Some(routes) = routes else {
        return Ok(None);
    };
    if exposed.is_none() {
        let listener = tokio
            .block_on(tokio::net::TcpListener::bind(address))
            .map_err(|e| format!("Cannot listen on {address}: {e}"))?;
        // Without a key the exposed API is open, as jevons-rs is without one.
        let forwarder = Forwarder::new(key.clone());
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let task = tokio.spawn(forwarder.clone().serve(listener, async {
            let _ = stopped.await;
        }));
        tracing::info!(%address, "Serving the API to other clients");
        *exposed = Some(Exposed {
            address,
            key,
            forwarder,
            stop,
            task,
        });
    }
    let serving = exposed.as_ref().expect("the forwarder listens");
    serving.forwarder.route(forward::targets(routes));
    Ok(Some(format!("http://{address}")))
}

struct Shared {
    status: Status,
    routes: Option<Routes>,
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
            routes: None,
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

    /// Each capability's route, once any provider answers.
    pub fn routes(&self) -> Option<Routes> {
        self.shared.lock().expect("the runtime lock").routes.clone()
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
    let set = |status: Status, routes: Option<Routes>| {
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
        shared.routes = routes;
        drop(shared);
        changed();
    };
    let mut embedded = embedded::Embedded::default();
    let mut exposed: Option<Exposed> = None;
    while let Ok(Some((config, file))) = requests.recv() {
        // Only the latest settings matter.
        let (config, file) = std::iter::from_fn(|| requests.try_recv().ok())
            .map_while(|r| r)
            .last()
            .unwrap_or((config, file));
        if let Err(e) = config.check_routes() {
            tokio.block_on(embedded.unload());
            let _ = expose(&tokio, &mut exposed, &config, None);
            set(Status::Failed(e), None);
            continue;
        }
        // What does not answer is said, and the capabilities on the providers that do still
        // run.
        let mut failures: Vec<String> = Vec::new();
        let mut served: BTreeMap<String, Served> = BTreeMap::new();
        let mut local = false;
        let providers = used(&config);
        // The models in this app: only those of the capabilities routed here are loaded.
        let wanted = Local::of(&config);
        let settings = if wanted.any() {
            let catalog_file = file.parent().unwrap_or(Path::new(".")).join("models.toml");
            let (catalog, _) = jevons_desktop_core::catalog::load(&catalog_file);
            let folder = config.models_folder();
            let models = crate::config::with_downloaded(&config.models, &folder, &catalog);
            runtime_settings(&models, &file, wanted).unwrap_or_else(|e| {
                failures.push(e);
                None
            })
        } else {
            None
        };
        match settings {
            Some(settings) => {
                if embedded.needs_load(&settings) {
                    set(Status::Loading, None);
                }
                match embedded.apply(&tokio, &settings, &file) {
                    Ok(server) => {
                        local = true;
                        for (name, provider) in &providers {
                            if provider.kind == ProviderKind::Embedded {
                                served.insert(name.clone(), server.clone());
                            }
                        }
                    }
                    Err(e) => failures.push(e),
                }
            }
            None => tokio.block_on(embedded.unload()),
        }
        // The servers elsewhere: a jevons server says which model serves each capability.
        let mut elsewhere = Vec::new();
        for (name, provider) in &providers {
            let Some(url) = provider.url() else {
                continue;
            };
            let client = Client::new(&url, provider.key());
            let server = if provider.kind == ProviderKind::Jevons {
                match tokio.block_on(client.health()) {
                    Ok(health) => Served {
                        speech: health.services.speech,
                        decision: health.services.decision,
                        generation: health.services.generative,
                        ..Served::unnamed(client)
                    },
                    Err(e) => {
                        failures.push(format!("{name} ({url}): {e}"));
                        continue;
                    }
                }
            } else {
                Served::unnamed(client)
            };
            elsewhere.push(format!("{name} ({url})"));
            served.insert(name.clone(), server);
        }
        let routes = routes(&config, &served);
        let any = routes.speech.is_some()
            || routes.realtime.is_some()
            || routes.decision.is_some()
            || routes.generation.is_some();
        let routes = any.then_some(routes);
        // Other clients get what the routes serve, on the address the settings expose.
        let api = expose(&tokio, &mut exposed, &config, routes.as_ref()).unwrap_or_else(|e| {
            failures.push(e);
            None
        });
        let status = if !failures.is_empty() {
            Status::Failed(failures.join("; "))
        } else if !any {
            Status::NoModels
        } else if local {
            Status::Ready { api }
        } else {
            Status::Remote {
                providers: elsewhere,
                api,
            }
        };
        set(status, routes);
    }
    let _ = expose(&tokio, &mut exposed, &DesktopConfig::default(), None);
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

/// The jevons-rs settings for the selected models of the capabilities `wanted` here, or `None`
/// when there is none. A `runtime_config` file is loaded as it is.
pub fn runtime_settings(
    models: &Models,
    file: &Path,
    wanted: Local,
) -> Result<Option<String>, String> {
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
    for (service, model, wanted_here) in [
        ("generative", &models.generative, wanted.generation),
        ("decision", &models.decision, wanted.decision),
        ("speech", &models.speech, wanted.speech),
    ] {
        if let Some(model) = model.as_ref().filter(|_| wanted_here) {
            let mut entry = toml::Table::new();
            entry.insert("model".into(), name_for(model).into());
            if service == "speech" {
                let realtime = models.realtime && wanted.realtime;
                entry.insert("realtime".into(), realtime.into());
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
    use super::Served;
    use jevons_api::config::Settings;
    use jevons_api::{AppState, Workers};
    use jevons_desktop_server::client::Client;
    use std::net::SocketAddr;
    use std::path::Path;
    use std::sync::Arc;
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;

    struct Loaded {
        settings: String,
        state: AppState,
        workers: Workers,
    }

    /// The embedded API's listener: a loopback port and a key only this app knows.
    struct Listener {
        address: SocketAddr,
        key: String,
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

        /// Loads the models of `settings` unless they are loaded, and serves them to this
        /// app alone. The listener keeps its port and key while the models stay.
        pub fn apply(
            &mut self,
            tokio: &tokio::runtime::Runtime,
            settings: &str,
            file: &Path,
        ) -> Result<Served, String> {
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
            let loaded = self.loaded.as_ref().expect("the models are loaded");
            if self.listener.is_none() {
                let key = format!(
                    "{}{}",
                    uuid::Uuid::new_v4().simple(),
                    uuid::Uuid::new_v4().simple()
                );
                let address = SocketAddr::from(([127, 0, 0, 1], 0));
                let listener = tokio
                    .block_on(tokio::net::TcpListener::bind(address))
                    .map_err(|e| format!("Cannot listen on {address}: {e}"))?;
                let address = listener.local_addr().map_err(|e| e.to_string())?;
                let state = AppState {
                    api_key: Some(Arc::from(key.as_str())),
                    ..loaded.state.clone()
                };
                let (stop, stopped) = oneshot::channel::<()>();
                let task = tokio.spawn(jevons_api::serve(listener, state, async {
                    let _ = stopped.await;
                }));
                tracing::info!(%address, "Serving the embedded API to this app");
                self.listener = Some(Listener {
                    address,
                    key,
                    stop,
                    task,
                });
            }
            let listener = self.listener.as_ref().expect("the listener runs");
            let state = &loaded.state;
            Ok(Served {
                client: Client::new(
                    &format!("http://{}", listener.address),
                    Some(listener.key.clone()),
                ),
                speech: state.speech.as_ref().map(|s| s.model_id.clone()),
                decision: state.decision.as_ref().map(|s| s.model_id.clone()),
                generation: state.generative.as_ref().map(|s| s.model_id.clone()),
                realtime: state.speech.as_ref().is_some_and(|s| s.realtime),
            })
        }
    }
}

#[cfg(not(feature = "embedded"))]
mod embedded {
    use super::Served;
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
        ) -> Result<Served, String> {
            Err(
                "this build has no embedded runtime: route every capability to another provider"
                    .into(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: Local = Local {
        speech: true,
        realtime: true,
        decision: true,
        generation: true,
    };

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
        let file = Path::new("/c/jevons-desktop.toml");
        let text = runtime_settings(&models, file, ALL).unwrap().unwrap();
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
            runtime_settings(&Models::default(), Path::new("x.toml"), ALL).unwrap(),
            None
        );
    }

    fn config(text: &str) -> DesktopConfig {
        let config: DesktopConfig = toml::from_str(text).unwrap();
        assert_eq!(config.check_routes(), Ok(()));
        config
    }

    const ELSEWHERE: &str = r#"
[providers.openrouter]
kind = "openrouter"
min_probability = 0.6

[providers.box]
kind = "jevons"

[routes]
decision = { provider = "openrouter", model = "typesafe/jev-1.13" }
generation = { provider = "box" }
"#;

    #[test]
    fn only_the_models_of_capabilities_routed_here_are_loaded() {
        // Decisions and generation go elsewhere: the language model is not loaded.
        let wanted = Local::of(&config(ELSEWHERE));
        assert_eq!(
            wanted,
            Local {
                speech: true,
                realtime: true,
                decision: false,
                generation: false
            }
        );
        let models = Models {
            generative: model("/m/gemma"),
            decision: model("/m/gemma"),
            speech: model("/m/parakeet"),
            ..Models::default()
        };
        let file = Path::new("/c/jevons-desktop.toml");
        let text = runtime_settings(&models, file, wanted).unwrap().unwrap();
        let table: toml::Table = toml::from_str(&text).unwrap();
        assert_eq!(table["models"].as_table().unwrap().len(), 1);
        let services = table["services"].as_table().unwrap();
        assert_eq!(services.keys().collect::<Vec<_>>(), ["speech"]);
        assert_eq!(services["speech"]["realtime"].as_bool(), Some(true));
        // With every capability elsewhere, nothing is.
        let none = Local::of(&config(
            "[providers.box]\nkind = \"jevons\"\n[routes]\nspeech = { provider = \"box\" }\n\
             decision = { provider = \"box\" }\ngeneration = { provider = \"box\" }\n",
        ));
        assert_eq!(none, Local::default());
        assert_eq!(runtime_settings(&models, file, none).unwrap(), None);
        // By default all of them are here.
        assert_eq!(Local::of(&DesktopConfig::default()), ALL);
    }

    #[test]
    fn exposing_the_api_serves_what_the_routes_serve_until_it_is_turned_off() {
        let tokio = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let free = || {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let mut config = DesktopConfig::default();
        config.server.expose = true;
        config.server.port = free();
        let route = |model: &str| Routes {
            decision: Some(Route::ours(
                Client::new("http://127.0.0.1:1", None),
                "box",
                model,
            )),
            ..Routes::default()
        };
        let health = |api: &str| tokio.block_on(Client::new(api, None).health());
        let mut exposed = None;
        // Private: nothing listens, and there is no address to give.
        config.server.expose = false;
        let api = expose(&tokio, &mut exposed, &config, Some(&route("jev"))).unwrap();
        assert!(api.is_none() && exposed.is_none());
        // Exposed: other clients see what the routes serve.
        config.server.expose = true;
        let api = expose(&tokio, &mut exposed, &config, Some(&route("jev")))
            .unwrap()
            .unwrap();
        assert_eq!(api, format!("http://127.0.0.1:{}", config.server.port));
        assert_eq!(
            health(&api).unwrap().services.decision.as_deref(),
            Some("jev")
        );
        // New routes are served by the same listener.
        expose(&tokio, &mut exposed, &config, Some(&route("gemma"))).unwrap();
        assert_eq!(
            health(&api).unwrap().services.decision.as_deref(),
            Some("gemma")
        );
        // Another port moves it.
        let old = api;
        config.server.port = free();
        let api = expose(&tokio, &mut exposed, &config, Some(&route("gemma")))
            .unwrap()
            .unwrap();
        assert!(health(&old).is_err());
        assert!(health(&api).is_ok());
        // An address that is taken says so.
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut elsewhere = config.clone();
        elsewhere.server.port = taken.local_addr().unwrap().port();
        let mut second = None;
        let error = expose(&tokio, &mut second, &elsewhere, Some(&route("jev"))).unwrap_err();
        assert!(error.starts_with("Cannot listen on 127.0.0.1:"), "{error}");
        // With nothing served, or turned off, it stops.
        assert_eq!(expose(&tokio, &mut exposed, &config, None), Ok(None));
        assert!(exposed.is_none() && health(&api).is_err());
    }

    #[test]
    fn each_route_takes_its_own_model_or_the_one_its_provider_names() {
        let client = |port: u16| Client::new(&format!("http://127.0.0.1:{port}"), None);
        let served = BTreeMap::from([
            (
                "embedded".to_string(),
                Served {
                    speech: Some("parakeet".into()),
                    decision: Some("gemma".into()),
                    generation: Some("gemma".into()),
                    realtime: true,
                    client: client(1),
                },
            ),
            ("openrouter".to_string(), Served::unnamed(client(2))),
            (
                "box".to_string(),
                Served {
                    generation: Some("nemotron".into()),
                    ..Served::unnamed(client(3))
                },
            ),
        ]);
        let config = config(ELSEWHERE);
        let all = routes(&config, &served);
        let named = |route: &Option<Route>| route.as_ref().map(Route::name);
        assert_eq!(named(&all.speech).as_deref(), Some("embedded/parakeet"));
        assert_eq!(named(&all.realtime).as_deref(), Some("embedded/parakeet"));
        assert_eq!(
            named(&all.decision).as_deref(),
            Some("openrouter/typesafe/jev-1.13")
        );
        assert_eq!(named(&all.generation).as_deref(), Some("box/nemotron"));
        assert_eq!(all.generation.unwrap().client.base(), "http://127.0.0.1:3");
        // Each route carries its provider's profile, with what the settings set of it.
        let decision = all.decision.unwrap().profile;
        assert_eq!(
            (
                decision.steps,
                decision.max_questions,
                decision.min_probability
            ),
            (false, Some(8), 0.6)
        );
        assert_eq!(all.speech.unwrap().profile, Profile::ours());
        // A provider that does not answer leaves its capabilities unserved, and so does one
        // that names no model for them; speech that does not stream leaves Realtime off.
        let mut partial = served.clone();
        partial.remove("openrouter");
        partial.get_mut("box").unwrap().generation = None;
        partial.get_mut("embedded").unwrap().realtime = false;
        let left = routes(&config, &partial);
        assert!(left.decision.is_none() && left.generation.is_none());
        assert!(left.speech.is_some() && left.realtime.is_none());
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
        let text = runtime_settings(&models, file, ALL).unwrap().unwrap();
        let settings = jevons_api::config::Settings::parse(&text, file).unwrap();
        assert!(settings.services.speech.unwrap().realtime);
    }
}
