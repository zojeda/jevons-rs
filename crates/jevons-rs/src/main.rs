use jevons_engine::{
    ModelConfig, SpeechConfig, default_model_id, default_speech_model_id, detect_speech,
    resolve_architecture,
};
use jevons_rs::{AppState, SpeechService, TextService, config::Settings, router, speech, worker};
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;

type Error = Box<dyn std::error::Error>;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Settings::load()?;
    if args.model.is_none() && args.speech_model.is_none() {
        return Err(
            "Set a language model with --model (DIFFUSION_MODEL) or a speech model with \
                    --speech-model (JEVONS_SPEECH_MODEL), or either in the settings file"
                .into(),
        );
    }
    if args.api_key.as_ref().is_some_and(|key| key.is_empty()) {
        return Err("The configured API key must be nonempty".into());
    }
    if !(args.max_audio_seconds.is_finite() && args.max_audio_seconds > 0.0) {
        return Err("--max-audio-seconds must be positive".into());
    }
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    let mut threads = Vec::new();
    let text = match args.model.clone() {
        Some(model) => Some(load_text(&args, model, &mut threads).await?),
        None => None,
    };
    let speech = match args.speech_model.clone() {
        Some(model) => Some(load_speech(&args, &model, &mut threads).await?),
        None => None,
    };
    if let (Some(text), Some(speech)) = (&text, &speech)
        && speech.names().iter().any(|name| text.serves(name))
    {
        return Err("The language and speech models must have different IDs and aliases".into());
    }
    let app = router(AppState {
        text,
        speech,
        api_key: args.api_key.map(Arc::from),
    });
    tracing::info!(address = %args.bind, "jevons-rs is ready");
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

async fn load_text(
    args: &Settings,
    model: std::path::PathBuf,
    threads: &mut Vec<JoinHandle<()>>,
) -> Result<TextService, Error> {
    let mut config = ModelConfig::new(model);
    config.architecture = Some(args.arch.clone());
    let architecture = resolve_architecture(&config)?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| default_model_id(&config.model, architecture));
    if model_id.trim().is_empty() {
        return Err("The model ID must be nonempty".into());
    }
    config.mmproj = args.mmproj.clone();
    if config.mmproj.is_some() && !architecture.uses_separate_projector() {
        // Often inherited from DIFFUSION_MMPROJ; the checkpoint carries its own vision tower.
        tracing::warn!(
            architecture = architecture.id(),
            "Ignoring --mmproj: this architecture's vision tower is in the model files"
        );
        config.mmproj = None;
    }
    config.main_gpu = args.main_gpu;
    config.context_size = args.context_size;
    config.batch_size = args.batch_size;
    config.prompt_cache = !args.no_prompt_cache;
    tracing::info!(architecture = architecture.id(), "Loading model");
    let (client, thread, info) = worker::start(
        config,
        model_id.clone(),
        args.seed,
        args.queue_capacity,
        args.decoding,
    )
    .await?;
    threads.push(thread);
    tracing::info!(
        model = %info.display_name,
        decoding = args.decoding.id(),
        "Model loaded"
    );
    Ok(TextService {
        worker: client,
        aliases: [architecture.latest_alias(), "openjev-latest", "jev-latest"]
            .into_iter()
            .filter(|alias| *alias != model_id)
            .map(String::from)
            .collect(),
        description: format!(
            "Local {}, structured diffusion reads. Served as {model_id}.",
            info.display_name
        ),
        model_id,
    })
}

async fn load_speech(
    args: &Settings,
    model: &Path,
    threads: &mut Vec<JoinHandle<()>>,
) -> Result<SpeechService, Error> {
    let architecture = detect_speech(model)?;
    let model_id = args
        .speech_model_id
        .clone()
        .unwrap_or_else(|| default_speech_model_id(model, architecture));
    if model_id.trim().is_empty() {
        return Err("The speech model ID must be nonempty".into());
    }
    let mut config = SpeechConfig::new(model);
    config.main_gpu = args.main_gpu;
    tracing::info!(architecture = architecture.id(), "Loading speech model");
    let (client, thread, info) = speech::start(config, args.speech_queue_capacity).await?;
    threads.push(thread);
    tracing::info!(
        model = %info.display_name,
        languages = info.languages.len(),
        realtime = !args.no_realtime,
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
            "Local {}, speech to text in {} languages. Served as {model_id}.",
            info.display_name,
            info.languages.len()
        ),
        model_id,
        info,
        max_audio_seconds: args.max_audio_seconds,
        realtime: !args.no_realtime,
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
