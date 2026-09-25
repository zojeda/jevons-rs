//! Startup: load each configured model once on its worker thread, attach the services that use
//! it, then serve the API.

use crate::config::{Model, Settings, Speech};
use crate::workers::{diffusion as worker, speech};
use crate::{AppState, DiffusionService, SpeechService, router};
use jevons_core::{ModelConfig, SpeechConfig};
use jevons_diffusion::{default_model_id, resolve_architecture};
use jevons_speech::{default_speech_model_id, detect_speech};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread::JoinHandle;

type Error = Box<dyn std::error::Error>;

/// Loads the configured models, starts their workers and serves until SIGINT or SIGTERM.
pub async fn run(settings: Settings) -> Result<(), Error> {
    let listener = tokio::net::TcpListener::bind(settings.server.bind).await?;
    let mut threads = Vec::new();
    let services = &settings.services;
    let diffusion_models = settings.diffusion_models();
    let mut engines: BTreeMap<&str, DiffusionService> = BTreeMap::new();
    for &name in &diffusion_models {
        let uses: Vec<&str> = [
            ("generative", &services.generative),
            ("decision", &services.decision),
        ]
        .into_iter()
        .filter(|(_, s)| s.as_ref().is_some_and(|s| s.model == name))
        .map(|(service, _)| service)
        .collect();
        // The generic `jev-latest` / `openjev-latest` aliases name the System One model, or the
        // only diffusion model.
        let generic = uses.contains(&"decision") || diffusion_models.len() == 1;
        let service =
            load_diffusion(name, &settings.models[name], &uses, generic, &mut threads).await?;
        engines.insert(name, service);
    }
    let engine = |service: &Option<crate::config::ServiceModel>| {
        service.as_ref().map(|s| engines[s.model.as_str()].clone())
    };
    let generative = engine(&services.generative);
    let decision = engine(&services.decision);
    let speech = match &services.speech {
        Some(speech) => Some(
            load_speech(
                &speech.model,
                &settings.models[&speech.model],
                speech,
                &mut threads,
            )
            .await?,
        ),
        None => None,
    };
    let mut names: Vec<String> = Vec::new();
    for service in engines.values() {
        names.extend(
            std::iter::once(service.model_id.clone()).chain(service.aliases.iter().cloned()),
        );
    }
    names.extend(speech.iter().flat_map(SpeechService::names));
    if let Some(name) = names
        .iter()
        .enumerate()
        .find_map(|(i, n)| names[..i].contains(n).then_some(n))
    {
        return Err(format!("Two models answer to {name:?}; give them different ids").into());
    }
    // The router owns the only queue handles from here on: when it is dropped at shutdown, the
    // workers see their queues close and their threads end.
    drop(engines);
    let app = router(AppState {
        generative,
        decision,
        speech,
        api_key: settings.server.api_key.clone().map(Arc::from),
    });
    tracing::info!(address = %settings.server.bind, "jevons-rs is ready");
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await;
    for thread in threads {
        tokio::task::spawn_blocking(move || thread.join())
            .await?
            .map_err(|_| "Inference worker panicked")?;
    }
    result?;
    Ok(())
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
