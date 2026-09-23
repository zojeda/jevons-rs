use clap::Parser;
use jevons_engine::ModelConfig;
use jevons_rs::{AppState, router, worker};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command(about = "A System One API backed by DiffusionGemma structured reads")]
struct Args {
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    /// Compatible DiffusionGemma vision projector; required for image requests.
    #[arg(long, env = "DIFFUSION_MMPROJ")]
    mmproj: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long, default_value = "gemmadiffusion-0.1")]
    model_id: String,
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
    if args.model_id.trim().is_empty() || args.api_key.as_ref().is_some_and(|key| key.is_empty()) {
        return Err("Model ID and configured API key must be nonempty".into());
    }
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    let mut config = ModelConfig::new(args.model);
    config.mmproj = args.mmproj;
    config.main_gpu = args.main_gpu;
    config.context_size = args.context_size;
    config.batch_size = args.batch_size;
    config.prompt_cache = !args.no_prompt_cache;
    tracing::info!("Loading DiffusionGemma");
    let (client, thread) = worker::start(
        config,
        args.model_id.clone(),
        args.seed,
        args.queue_capacity,
    )
    .await?;
    let app = router(AppState {
        worker: client,
        model_id: args.model_id,
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
