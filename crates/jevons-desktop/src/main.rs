//! jevons-desktop: context-aware dictation from the tray.
//!
//! Tap the hotkey to toggle dictation or hold it while speaking. The agent reads the focused
//! application and field, transcribes, picks a profile, decides whether to insert, replace or
//! rewrite, generates the text when it needs editing, and types it where the user was.
//!
//! `--replay <audio> --context <json>` runs one take headless and prints its trace.
#![forbid(unsafe_code)]

mod agent;
mod audio;
mod platform;
mod runtime;
mod tray;
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        // stdout carries the --replay trace.
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let config_file = args.config.clone().unwrap_or_else(default_config_file);
    let config = DesktopConfig::load(&config_file)?;
    match &args.replay {
        Some(audio) => replay(&args, audio, config, config_file),
        None => desktop(config, config_file),
    }
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
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("jevons")
            .with_inner_size([760.0, 640.0])
            .with_visible(false),
        ..Default::default()
    };
    let app_view = view.clone();
    let app_commands = commands.clone();
    eframe::run_native(
        "jevons",
        options,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(move || ctx.request_repaint());
            start_agent(
                config,
                config_file,
                app_view.clone(),
                repaint,
                app_commands.clone(),
                received,
            )?;
            Ok(Box::new(ui::App::new(app_view, app_commands)))
        }),
    )?;
    Ok(())
}

fn start_agent(
    config: DesktopConfig,
    config_file: PathBuf,
    view: agent::SharedView,
    repaint: Arc<dyn Fn() + Send + Sync>,
    commands: mpsc::UnboundedSender<Command>,
    received: mpsc::UnboundedReceiver<Command>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
