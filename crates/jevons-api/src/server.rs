//! Startup: load each configured model once on its worker thread, attach the services that use
//! it, then serve the API.

use crate::config::{Model, Settings, Speech, Voice};
use crate::workers::{diffusion as worker, speech, voice};
use crate::{AppState, DiffusionService, SpeechService, VoiceService, router};
use jevons_core::{ModelConfig, SpeechConfig, VoiceConfig};
use jevons_diffusion::{default_model_id, resolve_architecture};
use jevons_speech::{default_speech_model_id, detect_speech};
use jevons_voice::{default_voice_model_id, detect_voice};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread::JoinHandle;

type Error = Box<dyn std::error::Error>;

/// Loads the configured models, starts their workers and serves until SIGINT or SIGTERM.
pub async fn run(settings: Settings) -> Result<(), Error> {
    let listener = tokio::net::TcpListener::bind(settings.server.bind).await?;
    let (state, workers) = load(&settings).await?;
    tracing::info!(address = %settings.server.bind, "jevons-rs is ready");
    let result = serve(listener, state, shutdown()).await;
    workers.join().await?;
    result?;
    Ok(())
}

/// The threads that own the loaded models. They end once every [`AppState`] clone holding their
/// queues is dropped.
#[must_use = "join the workers after the last AppState is dropped"]
pub struct Workers {
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    /// Whether a model thread has ended. While an [`AppState`] clone holds its queues, a worker
    /// ends only when its model failed, such as by a panic on a GPU that stopped answering.
    pub fn stopped(&self) -> bool {
        self.threads.iter().any(JoinHandle::is_finished)
    }

    /// Waits for the model threads to end; drop every [`AppState`] clone first. Every thread
    /// is waited for, also after one panicked: a caller that goes on to exit must not leave a
    /// model still being freed on the device.
    pub async fn join(self) -> Result<(), Error> {
        let mut panicked = false;
        for thread in self.threads {
            let ended = tokio::task::spawn_blocking(move || thread.join()).await;
            panicked |= !matches!(ended, Ok(Ok(())));
        }
        if panicked {
            return Err("Inference worker panicked".into());
        }
        Ok(())
    }
}

/// Serves the API on `listener` until `shutdown` completes, then finishes pending requests.
/// The models stay loaded: serve the same `state` again on another listener to rebind.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: AppState,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}

/// A service of the settings file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Service {
    Generative,
    Decision,
    Speech,
    Voice,
}

impl Service {
    /// The service's key under `[services]`, and in `/health`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Generative => "generative",
            Self::Decision => "decision",
            Self::Speech => "speech",
            Self::Voice => "voice",
        }
    }
}

/// A service [`load_optional`] left unserved, and why its model did not load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unserved {
    pub service: Service,
    pub error: String,
}

/// What [`load_optional`] loaded.
#[must_use = "join the workers after the last AppState is dropped"]
pub struct Loaded {
    pub state: AppState,
    pub workers: Workers,
    /// The optional services whose model failed to load. The state serves the others, and
    /// `/health` reports these as off.
    pub unserved: Vec<Unserved>,
}

/// Loads the configured models once and starts their workers. `server.api_key` becomes the
/// state's key; override it per listener with struct update syntax.
///
/// When a model fails to load, those loaded before it are dropped, and this returns once their
/// workers have ended: the caller may exit, or load again, with none still being freed on the
/// device.
pub async fn load(settings: &Settings) -> Result<(AppState, Workers), Error> {
    let loaded = load_optional(settings, &[]).await?;
    Ok((loaded.state, loaded.workers))
}

/// [`load`], except that the services in `optional` may fail: when a model fails to load and
/// every service that names it is optional, the others are served without it, and the failure
/// comes back in [`Loaded::unserved`] for the caller to show. Any other failure is an error, as
/// for [`load`]. The voice model loads last.
pub async fn load_optional(settings: &Settings, optional: &[Service]) -> Result<Loaded, Error> {
    let mut threads = Vec::new();
    let mut unserved = Vec::new();
    match load_into(settings, optional, &mut threads, &mut unserved).await {
        Ok(state) => Ok(Loaded {
            state,
            workers: Workers { threads },
            unserved,
        }),
        Err(error) => {
            // The queues of what did load went with the failed attempt: their workers end.
            let _ = Workers { threads }.join().await;
            Err(error)
        }
    }
}

/// What loading a model gave the services in `uses`: the loaded model, or nothing when it
/// failed and every one of those services is optional. The failure is then kept in `unserved`,
/// once per service.
fn served<T>(
    loaded: Result<T, Error>,
    uses: &[Service],
    optional: &[Service],
    unserved: &mut Vec<Unserved>,
) -> Result<Option<T>, Error> {
    match loaded {
        Ok(service) => Ok(Some(service)),
        Err(error) if uses.iter().all(|service| optional.contains(service)) => {
            for &service in uses {
                tracing::warn!(
                    service = service.name(),
                    error = %error,
                    "Serving without an optional service: its model did not load"
                );
                unserved.push(Unserved {
                    service,
                    error: error.to_string(),
                });
            }
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

async fn load_into(
    settings: &Settings,
    optional: &[Service],
    threads: &mut Vec<JoinHandle<()>>,
    unserved: &mut Vec<Unserved>,
) -> Result<AppState, Error> {
    let services = &settings.services;
    let diffusion_models = settings.diffusion_models();
    let mut engines: BTreeMap<&str, DiffusionService> = BTreeMap::new();
    for &name in &diffusion_models {
        let uses: Vec<Service> = [
            (Service::Generative, &services.generative),
            (Service::Decision, &services.decision),
        ]
        .into_iter()
        .filter(|(_, s)| s.as_ref().is_some_and(|s| s.model == name))
        .map(|(service, _)| service)
        .collect();
        // The generic `jev-latest` / `openjev-latest` aliases name the System One model, or the
        // only diffusion model.
        let generic = uses.contains(&Service::Decision) || diffusion_models.len() == 1;
        let names: Vec<&str> = uses.iter().map(|service| service.name()).collect();
        let loaded = load_diffusion(name, &settings.models[name], &names, generic, threads).await;
        if let Some(service) = served(loaded, &uses, optional, unserved)? {
            engines.insert(name, service);
        }
    }
    let engine = |service: &Option<crate::config::ServiceModel>| {
        service
            .as_ref()
            .and_then(|s| engines.get(s.model.as_str()).cloned())
    };
    let generative = engine(&services.generative);
    let decision = engine(&services.decision);
    let speech = match &services.speech {
        Some(speech) => {
            let model = &settings.models[&speech.model];
            let loaded = load_speech(&speech.model, model, speech, threads).await;
            served(loaded, &[Service::Speech], optional, unserved)?
        }
        None => None,
    };
    // Last, so that a voice model that fails has cost the other services nothing.
    let voice = match &services.voice {
        Some(voice) => {
            let model = &settings.models[&voice.model];
            let loaded = load_voice(&voice.model, model, voice, threads).await;
            served(loaded, &[Service::Voice], optional, unserved)?
        }
        None => None,
    };
    let mut names: Vec<String> = Vec::new();
    for service in engines.values() {
        names.extend(
            std::iter::once(service.model_id.clone()).chain(service.aliases.iter().cloned()),
        );
    }
    names.extend(speech.iter().flat_map(SpeechService::names));
    names.extend(voice.iter().flat_map(VoiceService::names));
    if let Some(name) = names
        .iter()
        .enumerate()
        .find_map(|(i, n)| names[..i].contains(n).then_some(n))
    {
        return Err(format!("Two models answer to {name:?}; give them different ids").into());
    }
    // The state owns the only queue handles from here on: when its last clone is dropped at
    // shutdown, the workers see their queues close and their threads end.
    drop(engines);
    Ok(AppState {
        generative,
        decision,
        speech,
        voice,
        api_key: settings.server.api_key.clone().map(Arc::from),
    })
}

/// Loads a diffusion language model once for the services in `uses`.
async fn load_diffusion(
    name: &str,
    model: &Model,
    uses: &[&str],
    generic_aliases: bool,
    threads: &mut Vec<JoinHandle<()>>,
) -> Result<DiffusionService, Error> {
    let mut config = ModelConfig::new(model.path.clone());
    config.architecture = model.arch.clone();
    let architecture = resolve_architecture(&config).map_err(|e| format!("models.{name}: {e}"))?;
    let model_id = model
        .id
        .clone()
        .unwrap_or_else(|| default_model_id(&config.model, architecture));
    config.mmproj = model.mmproj.clone();
    if config.mmproj.is_some() && !architecture.uses_separate_projector() {
        tracing::warn!(
            model = name,
            architecture = architecture.id(),
            "Ignoring mmproj: this architecture's vision tower is in the model files"
        );
        config.mmproj = None;
    }
    config.main_gpu = model.main_gpu;
    config.context_size = model.context_size();
    config.batch_size = model.batch_size();
    config.prompt_cache = model.prompt_cache();
    let decoding = model.decoding()?;
    tracing::info!(model = name, architecture = architecture.id(), services = ?uses, "Loading model");
    let (client, thread, info) = worker::start(
        config,
        model_id.clone(),
        model.seed(),
        model.queue_capacity,
        decoding,
    )
    .await?;
    threads.push(thread);
    tracing::info!(model = %info.display_name, decoding = decoding.id(), "Model loaded");
    let generic: &[&str] = if generic_aliases {
        &["openjev-latest", "jev-latest"]
    } else {
        &[]
    };
    Ok(DiffusionService {
        worker: client,
        aliases: std::iter::once(architecture.latest_alias())
            .chain(generic.iter().copied())
            .filter(|alias| *alias != model_id)
            .map(String::from)
            .collect(),
        description: format!(
            "Local {}, serving {}. Served as {model_id}.",
            info.display_name,
            uses.join(" and ")
        ),
        model_id,
    })
}

async fn load_speech(
    name: &str,
    model: &Model,
    service: &Speech,
    threads: &mut Vec<JoinHandle<()>>,
) -> Result<SpeechService, Error> {
    let architecture = detect_speech(&model.path).map_err(|e| format!("models.{name}: {e}"))?;
    let options = model.diffusion_options();
    if !options.is_empty() {
        return Err(format!(
            "models.{name} is a speech model; {} apply to diffusion language models only",
            options.join(", ")
        )
        .into());
    }
    let model_id = model
        .id
        .clone()
        .unwrap_or_else(|| default_speech_model_id(&model.path, architecture));
    let mut config = SpeechConfig::new(model.path.clone());
    config.main_gpu = model.main_gpu;
    tracing::info!(
        model = name,
        architecture = architecture.id(),
        "Loading speech model"
    );
    let (client, thread, info) = speech::start(config, model.queue_capacity).await?;
    threads.push(thread);
    tracing::info!(
        model = %info.display_name,
        languages = info.languages.len(),
        realtime = service.realtime,
        "Speech model loaded"
    );
    Ok(SpeechService {
        worker: client,
        aliases: [architecture.latest_alias()]
            .into_iter()
            .filter(|alias| *alias != model_id)
            .map(String::from)
            .collect(),
        description: format!(
            "Local {}, serving speech to text in {} languages. Served as {model_id}.",
            info.display_name,
            info.languages.len()
        ),
        model_id,
        info,
        max_audio_seconds: service.max_audio_seconds,
        realtime: service.realtime,
    })
}

async fn load_voice(
    name: &str,
    model: &Model,
    service: &Voice,
    threads: &mut Vec<JoinHandle<()>>,
) -> Result<VoiceService, Error> {
    let architecture = detect_voice(&model.path).map_err(|e| format!("models.{name}: {e}"))?;
    let options = model.diffusion_options();
    if !options.is_empty() {
        return Err(format!(
            "models.{name} is a voice model; {} apply to diffusion language models only",
            options.join(", ")
        )
        .into());
    }
    let model_id = model
        .id
        .clone()
        .unwrap_or_else(|| default_voice_model_id(&model.path, architecture));
    let mut config = VoiceConfig::new(model.path.clone());
    config.main_gpu = model.main_gpu;
    tracing::info!(
        model = name,
        architecture = architecture.id(),
        "Loading voice model"
    );
    let (client, thread, ready) =
        voice::start(config, service.voice.clone(), model.queue_capacity).await?;
    threads.push(thread);
    let info = ready.info;
    tracing::info!(
        model = %info.display_name,
        sample_rate = info.sample_rate,
        languages = info.languages.len(),
        voices = info.voices.len(),
        "Voice model loaded"
    );
    Ok(VoiceService {
        worker: client,
        aliases: [architecture.latest_alias()]
            .into_iter()
            .filter(|alias| *alias != model_id)
            .map(String::from)
            .collect(),
        description: format!(
            "Local {}, serving text to speech in {} languages and {} voices. Served as {model_id}.",
            info.display_name,
            info.languages.len(),
            info.voices.len()
        ),
        model_id,
        info,
        default_voice: ready.default_voice,
        max_input_chars: service.max_input_chars,
    })
}

async fn shutdown() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Cannot install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Stopping after pending requests finish");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::{mpsc, oneshot};

    #[tokio::test]
    async fn a_worker_that_ends_while_serving_shows_as_stopped() {
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        let workers = Workers {
            threads: vec![
                std::thread::spawn(move || {
                    let _ = stopped.recv();
                }),
                std::thread::spawn(|| std::thread::sleep(std::time::Duration::from_secs(5))),
            ],
        };
        assert!(!workers.stopped());
        drop(stop);
        let started = std::time::Instant::now();
        while !workers.stopped() {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(2),
                "a worker that ended still shows as running"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[tokio::test]
    async fn joining_waits_for_every_worker_even_after_one_panicked() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let ended = Arc::new(AtomicBool::new(false));
        let slow = ended.clone();
        let workers = Workers {
            threads: vec![
                std::thread::spawn(|| panic!("a worker that fails")),
                // A model still leaving the device when the first worker is found dead.
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(150));
                    slow.store(true, Ordering::SeqCst);
                }),
            ],
        };
        assert!(workers.join().await.is_err());
        assert!(
            ended.load(Ordering::SeqCst),
            "join returned while a worker still ran"
        );
    }

    fn state(key: Option<&str>) -> (AppState, mpsc::Receiver<worker::Job>) {
        let (sender, receiver) = mpsc::channel(1);
        let local = DiffusionService {
            worker: worker::Client { sender },
            model_id: "local".into(),
            aliases: Arc::from([]),
            description: "Local test model.".into(),
        };
        let state = AppState {
            generative: Some(local.clone()),
            decision: Some(local),
            speech: None,
            voice: None,
            api_key: key.map(Arc::from),
        };
        (state, receiver)
    }

    async fn get(address: std::net::SocketAddr, path: &str, key: Option<&str>) -> String {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let auth = key.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: test\r\n{auth}Connection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn rebinding_the_listener_keeps_models_loaded() {
        let (state, mut receiver) = state(None);
        let first = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let first_address = first.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        let server = tokio::spawn(serve(first, state.clone(), async {
            let _ = stopped.await;
        }));
        assert!(
            get(first_address, "/health", None)
                .await
                .contains("\"local\"")
        );
        stop.send(()).unwrap();
        server.await.unwrap().unwrap();
        assert!(tokio::net::TcpStream::connect(first_address).await.is_err());

        let second = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let second_address = second.local_addr().unwrap();
        let server = tokio::spawn(serve(second, state, std::future::pending()));
        assert!(
            get(second_address, "/health", None)
                .await
                .contains("\"local\"")
        );
        // The worker queue is still open: a live state holds its sender.
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        server.abort();
    }

    async fn health(state: AppState) -> serde_json::Value {
        use tower::ServiceExt;
        let request = axum::http::Request::builder()
            .uri("/health")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = router(state).oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn a_failed_model_is_left_out_only_when_every_service_that_names_it_is_optional() {
        use Service::{Decision, Generative, Speech, Voice};
        let failed = || -> Result<DiffusionService, Error> { Err("no such file".into()) };
        let mut unserved = Vec::new();
        // One of the two services of a shared model is required: the load fails.
        let error = served(
            failed(),
            &[Generative, Decision],
            &[Decision, Voice],
            &mut unserved,
        )
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "no such file");
        assert!(served(failed(), &[Speech], &[], &mut unserved).is_err());
        assert!(unserved.is_empty());
        // Both are optional: the model is left out, and each service says why.
        let left_out = served(
            failed(),
            &[Generative, Decision],
            &[Generative, Decision],
            &mut unserved,
        )
        .unwrap();
        assert!(left_out.is_none());
        let why = |service| Unserved {
            service,
            error: "no such file".into(),
        };
        assert_eq!(unserved, [why(Generative), why(Decision)]);
        // A model that loads is served, optional or not.
        let (state, _receiver) = state(None);
        let local = state.generative.clone().unwrap();
        let kept = served(Ok(local), &[Generative], &[Generative], &mut unserved).unwrap();
        assert!(kept.is_some());
        assert_eq!(unserved.len(), 2);
        // An optional voice model that fails leaves the loaded services up, and the health
        // check reports voice as off.
        let mut unserved = Vec::new();
        let voice: Result<VoiceService, Error> = Err("the device ran out of memory".into());
        let state = AppState {
            voice: served(voice, &[Voice], &[Voice], &mut unserved).unwrap(),
            ..state
        };
        assert_eq!(
            unserved,
            [Unserved {
                service: Voice,
                error: "the device ran out of memory".into(),
            }]
        );
        let health = health(state).await;
        assert_eq!(
            health["services"],
            serde_json::json!({
                "generative": "local", "decision": "local", "speech": null, "voice": null,
            })
        );
        assert_eq!(health["voice"], serde_json::Value::Null);
        assert_eq!(Voice.name(), "voice");
    }

    /// Settings whose only service is voice, on a folder that holds no checkpoint.
    fn settings_without_a_voice_checkpoint(name: &str) -> Settings {
        let dir = std::env::temp_dir().join(format!("jevons-load-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let text = "[models.qwen]\npath = \"missing\"\n[services.voice]\nmodel = \"qwen\"\n";
        Settings::parse(text, &dir.join("jevons.toml")).unwrap()
    }

    #[tokio::test]
    async fn a_voice_model_that_fails_to_load_stops_the_load_unless_voice_is_optional() {
        let settings = settings_without_a_voice_checkpoint("voice");
        // `load`, which `run` calls, names no optional service: it refuses.
        let error = load(&settings).await.err().unwrap();
        assert!(error.to_string().starts_with("models.qwen:"), "{error}");
        // Another service being optional changes nothing for voice.
        let refused = load_optional(&settings, &[Service::Speech]).await;
        assert!(refused.is_err());
        // With voice optional, the state comes back with the error beside it.
        let loaded = load_optional(&settings, &[Service::Voice]).await.unwrap();
        assert!(loaded.state.voice.is_none());
        assert_eq!(loaded.unserved.len(), 1);
        assert_eq!(loaded.unserved[0].service, Service::Voice);
        assert_eq!(loaded.unserved[0].error, error.to_string());
        let health = health(loaded.state).await;
        assert_eq!(health["services"]["voice"], serde_json::Value::Null);
        assert_eq!(health["voice"], serde_json::Value::Null);
        // No model loaded, so no worker is left to wait for.
        loaded.workers.join().await.unwrap();
    }

    #[tokio::test]
    async fn a_listener_can_override_the_api_key() {
        let (state, _receiver) = state(Some("from-settings"));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let state = AppState {
            api_key: Some(Arc::from("per-launch")),
            ..state
        };
        let server = tokio::spawn(serve(listener, state, std::future::pending()));
        let rejected = get(address, "/v1/models", Some("from-settings")).await;
        assert!(rejected.starts_with("HTTP/1.1 401"), "{rejected}");
        let accepted = get(address, "/v1/models", Some("per-launch")).await;
        assert!(accepted.starts_with("HTTP/1.1 200"), "{accepted}");
        server.abort();
    }
}
