use jevons_engine::{ModelConfig, default_model_id, resolve_architecture};
use jevons_rs::{AppState, config::Settings, router, worker};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Settings::load()?;
    let model = args
        .model
        .clone()
        .ok_or("Set the model with --model, DIFFUSION_MODEL or `model` in the settings file")?;
    let mut config = ModelConfig::new(model);
    config.architecture = Some(args.arch);
    let architecture = resolve_architecture(&config)?;
    let model_id = args
        .model_id
        .unwrap_or_else(|| default_model_id(&config.model, architecture));
    if model_id.trim().is_empty() || args.api_key.as_ref().is_some_and(|key| key.is_empty()) {
        return Err("Model ID and configured API key must be nonempty".into());
    }
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    config.mmproj = args.mmproj;
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
    tracing::info!(
        model = %info.display_name,
        decoding = args.decoding.id(),
        "Model loaded"
    );
    let app = router(AppState {
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
        api_key: args.api_key.map(Arc::from),
    });
    tracing::info!(address = %args.bind, "System One service is ready");
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await;
    tokio::task::spawn_blocking(move || thread.join())
        .await?
        .map_err(|_| "Inference worker panicked")?;
    result?;
    Ok(())
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
