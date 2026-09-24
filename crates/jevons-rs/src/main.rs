use clap::Parser;
use jevons_engine::{ModelConfig, resolve_architecture};
use jevons_rs::{AppState, router, worker};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command(about = "A System One API backed by structured diffusion-model reads")]
struct Args {
    /// GGUF file or Hugging Face checkpoint directory.
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    /// Model architecture; detected from the model files by default.
    #[arg(long, env = "JEVONS_ARCH", default_value = "auto")]
    arch: String,
    /// Separate vision projector for DiffusionGemma; required for its image requests.
    #[arg(long, env = "DIFFUSION_MMPROJ")]
    mmproj: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    /// Served model ID; defaults to the architecture's ID, such as gemmadiffusion-0.1.
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long, env = "TYPESAFE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// HIP device index.
    #[arg(long, default_value_t = 0)]
    main_gpu: usize,
    #[arg(long, default_value_t = 8192)]
    context_size: u32,
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    /// Recompute every prompt instead of reusing the longest cached token prefix.
    #[arg(long)]
    no_prompt_cache: bool,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    queue_capacity: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let mut config = ModelConfig::new(args.model);
    config.architecture = Some(args.arch);
    let architecture = resolve_architecture(&config)?;
    let model_id = args
        .model_id
        .unwrap_or_else(|| architecture.default_model_id().into());
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
    let (client, thread, info) =
        worker::start(config, model_id.clone(), args.seed, args.queue_capacity).await?;
    tracing::info!(model = %info.display_name, "Model loaded");
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
