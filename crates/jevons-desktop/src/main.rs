//! jevons-desktop: context-aware dictation from the tray.
//!
//! Tap the hotkey to toggle dictation or hold it while speaking. The agent reads the focused
//! application and field, transcribes, picks a profile, decides whether to insert, replace or
//! rewrite, generates the text when it needs editing, and types it where the user was.
//!
//! `--replay <audio> --context <json>` runs one take headless and prints its trace.
#![forbid(unsafe_code)]
// A tray app: no console window on Windows. Logs go to a file; --replay output can be redirected.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod agent;
mod audio;
mod hold;
mod platform;
mod runtime;
mod tray;
mod tuning;
mod ui;

use agent::{Agent, Command, Layers, View};
use clap::Parser;
use jevons_desktop_core::config::{DesktopConfig, default_config_file};
use jevons_desktop_core::context::ContextSnapshot;
use jevons_desktop_core::fake::FileAudioSource;
use jevons_desktop_core::pipeline::{self, Env, TakeStart};
use jevons_desktop_core::platform::AudioSource;
use jevons_desktop_core::profile::Profiles;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

#[derive(Parser)]
#[command(version, about = "Context-aware dictation over the jevons runtime")]
struct Args {
    /// The settings file (default: the platform configuration folder).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Run one take from an audio file instead of the microphone, print its trace and exit.
    #[arg(long, value_name = "AUDIO")]
    replay: Option<PathBuf>,
    /// The context for --replay, as a snapshot JSON file.
    #[arg(long, value_name = "JSON", requires = "replay")]
    context: Option<PathBuf>,
    /// With --replay: deliver into the focused application instead of only printing.
    #[arg(long, requires = "replay")]
    deliver: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let filter = || {
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "info,wgpu_hal=warn,wgpu_core=warn,naga=warn".into())
    };
    {
        use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
        let output = match args.replay.is_none().then(log_file).flatten() {
            // The tray app has no terminal: log to a file the user can find.
            Some(file) => tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(Mutex::new(file))
                .with_filter(filter())
                .boxed(),
            // stdout carries the --replay trace.
            None => tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(filter())
                .boxed(),
        };
        tracing_subscriber::registry()
            .with(output)
            .with(tuning::Layer)
            .init();
    }
    // Without a console, a panic on a worker thread would vanish: log it.
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!(thread = thread.name().unwrap_or("unnamed"), %info, %backtrace, "Panic");
    }));
    let config_file = args.config.clone().unwrap_or_else(default_config_file);
    let config = DesktopConfig::load(&config_file)?;
    match &args.replay {
        Some(audio) => replay(&args, audio, config, config_file),
        None => desktop(config, config_file),
    }
}

/// `~/jevons/logs/jevons-desktop.log`; the previous run's log is kept next to it.
fn log_file() -> Option<std::fs::File> {
    let dir = jevons_desktop_core::config::user_dir().join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let file = dir.join("jevons-desktop.log");
    let _ = std::fs::rename(&file, dir.join("jevons-desktop.previous.log"));
    std::fs::File::create(file).ok()
}

/// One headless take, for scripted end-to-end checks.
fn replay(
    args: &Args,
    audio: &std::path::Path,
    config: DesktopConfig,
    config_file: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let context: ContextSnapshot = match &args.context {
        Some(file) => serde_json::from_str(&std::fs::read_to_string(file)?)?,
        None => ContextSnapshot::default(),
    };
    let (ready, changed) = std::sync::mpsc::channel();
    let runtime = runtime::Runtime::start(move || {
        let _ = ready.send(());
    });
    runtime.apply(&config, &config_file);
    let connection = loop {
        changed.recv()?;
        match runtime.status() {
            runtime::Status::Loading => continue,
            status => match runtime.connection() {
                Some(connection) => break connection,
                None => return Err(status.describe().into()),
            },
        }
    };
    let profiles = Profiles::load_dir(&config.profiles_dir(&config_file));
    for error in &profiles.errors {
        eprintln!("{}: {}", error.file.display(), error.message);
    }
    let dictation = &config.dictation;
    let env = Env {
        client: connection.client,
        profiles: Arc::new(profiles),
        settings: pipeline::Settings {
            models: connection.models,
            realtime: connection.realtime,
            language: dictation.language.clone(),
            decide: dictation.decide,
            generation_threshold: dictation.generation_threshold,
            max_output_tokens: dictation.max_output_tokens,
            live_stream: dictation.live_stream,
            ..pipeline::Settings::default()
        },
        sink: args
            .deliver
            .then(|| Arc::new(Mutex::new(platform::text_sink()))),
    };
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let trace = tokio.block_on(async {
        let (events, received) = mpsc::unbounded_channel();
        let mut source = FileAudioSource {
            file: audio.to_path_buf(),
            paced: true,
        };
        let capture = source.start(None, events)?;
        let (_finish, finished) = oneshot::channel();
        let (updates, mut live) = mpsc::unbounded_channel();
        let printer = tokio::spawn(async move {
            while let Some(update) = live.recv().await {
                if let pipeline::Update::Delta(text) = update {
                    eprint!("{text}");
                }
            }
            eprintln!();
        });
        let start = TakeStart {
            id: 1,
            context,
            forced_profile: None,
        };
        let trace = pipeline::run_take(&env, start, received, finished, &updates).await;
        capture.stop();
        drop(updates);
        let _ = printer.await;
        Ok::<_, Box<dyn std::error::Error>>(trace)
    })?;
    println!("{}", serde_json::to_string_pretty(&trace)?);
    runtime.shutdown();
    match trace.error {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}

/// The tray, the agent and the inspector window.
fn desktop(config: DesktopConfig, config_file: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let view = Arc::new(Mutex::new(View::default()));
    let (commands, received) = mpsc::unbounded_channel::<Command>();
    let agent_view = view.clone();
    let agent_commands = commands.clone();
    ui::run(view, commands, move || {
        let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(ui::wake);
        if let Err(e) = start_agent(
            config,
            config_file,
            agent_view,
            repaint,
            agent_commands,
            received,
        ) {
            tracing::error!(error = %e, "Cannot start the agent");
        }
    })
}

fn start_agent(
    config: DesktopConfig,
    config_file: PathBuf,
    view: agent::SharedView,
    repaint: Arc<dyn Fn() + Send + Sync>,
    commands: mpsc::UnboundedSender<Command>,
    received: mpsc::UnboundedReceiver<Command>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    hold::start();
    let tray = match tray::spawn(commands.clone()) {
        Ok(tray) => {
            view.lock().expect("the view lock").tray_running = true;
            Some(tray)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Running without a tray icon");
            view.lock().expect("the view lock").show_window = true;
            None
        }
    };
    let runtime_commands = commands.clone();
    let runtime = runtime::Runtime::start(move || {
        let _ = runtime_commands.send(Command::RuntimeChanged);
    });
    std::thread::Builder::new()
        .name("agent".into())
        .spawn(move || {
            let tokio = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("agent-io")
                .build()
                .expect("the Tokio runtime builds");
            tokio.block_on(async move {
                let layers = Layers {
                    context: platform::context_provider(),
                    sink: platform::text_sink(),
                    audio: Box::new(audio::CpalSource),
                    tray,
                };
                let agent = Agent::new(
                    config,
                    config_file,
                    layers,
                    runtime,
                    view,
                    repaint,
                    commands,
                );
                agent.run(received).await;
            });
        })?;
    Ok(())
}
