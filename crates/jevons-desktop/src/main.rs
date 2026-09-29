//! jevons-desktop: a context-aware desktop agent in the tray.
//!
//! Tap the hotkey to toggle dictation or hold it while speaking. The agent reads the focused
//! application and field, transcribes, and walks the flow tree (a folder of TOML files) to a
//! leaf that types into the application, answers in the bubble or calls a tool.
//!
//! `--replay <audio>` or `--transcript <text>`, with `--context <json>`, runs one take headless
//! and prints its trace. `--check-flows` and `--init-flows` check and prepare a flows folder.
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
use jevons_desktop_core::flow::{FlowTree, defaults};
use jevons_desktop_core::pipeline::{self, Env, TakeStart};
use jevons_desktop_core::platform::AudioSource;
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
    #[arg(long, value_name = "AUDIO", conflicts_with = "transcript")]
    replay: Option<PathBuf>,
    /// Run one take from this text as if it had been said, print its trace and exit.
    #[arg(long, value_name = "TEXT")]
    transcript: Option<String>,
    /// The context for --replay or --transcript, as a snapshot JSON file.
    #[arg(long, value_name = "JSON")]
    context: Option<PathBuf>,
    /// With --replay or --transcript: start at this branch of the flow tree, such as `ask`.
    #[arg(long, value_name = "BRANCH")]
    flow: Option<String>,
    /// With --replay or --transcript: deliver into the focused application instead of only
    /// printing.
    #[arg(long)]
    deliver: bool,
    /// With --replay or --transcript: answer investigations from this recorded interface (the
    /// inspector's Record tree) instead of the live one.
    #[arg(long, value_name = "JSON")]
    tree: Option<PathBuf>,
    /// Check a flows folder (by default the settings' one), print its problems and exit.
    #[arg(long, value_name = "DIR", num_args = 0..=1, default_missing_value = "")]
    check_flows: Option<PathBuf>,
    /// Write the built-in flow tree into a folder that has none (by default the settings'
    /// flows folder), refresh AGENTS.md and the schemas, and exit.
    #[arg(long, value_name = "DIR", num_args = 0..=1, default_missing_value = "")]
    init_flows: Option<PathBuf>,
}

impl Args {
    fn headless(&self) -> bool {
        self.replay.is_some()
            || self.transcript.is_some()
            || self.check_flows.is_some()
            || self.init_flows.is_some()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let filter = || {
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "info,wgpu_hal=warn,wgpu_core=warn,naga=warn".into())
    };
    {
        use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
        let output = match (!args.headless()).then(log_file).flatten() {
            // The tray app has no terminal: log to a file the user can find.
            Some(file) => tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(Mutex::new(file))
                .with_filter(filter())
                .boxed(),
            // stdout carries the headless output.
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
    let folder = |dir: &PathBuf| {
        if dir.as_os_str().is_empty() {
            config.flows_dir(&config_file)
        } else {
            dir.clone()
        }
    };
    if let Some(dir) = &args.check_flows {
        return check_flows(&folder(dir), &config);
    }
    if let Some(dir) = &args.init_flows {
        let dir = folder(dir);
        let report = defaults::init(&dir)?;
        for file in &report.written {
            println!("wrote {}", dir.join(file).display());
        }
        for note in &report.notes {
            println!("note: {note}");
        }
        return check_flows(&dir, &config);
    }
    if args.replay.is_some() || args.transcript.is_some() {
        return replay(&args, config, config_file);
    }
    desktop(config, config_file)
}

/// Prints every problem of the flow tree in `dir`; fails when there is one.
fn check_flows(
    dir: &std::path::Path,
    config: &DesktopConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Tool names are checked against the settings; MCP tools by server (they are not started).
    let catalog =
        jevons_desktop_core::flow::tools::ToolHost::new(&config.tools, &config.mcp).catalog();
    let tree = FlowTree::load(&jevons_desktop_core::flow::Disk::new(dir), &catalog);
    for error in &tree.errors {
        eprintln!("{error}");
    }
    if !tree.is_valid() {
        return Err(format!(
            "{}: {} problem(s) in the flow tree",
            dir.display(),
            tree.errors.len()
        )
        .into());
    }
    let leaves = tree
        .nodes()
        .iter()
        .filter(|n| n.children.is_empty())
        .count();
    println!(
        "{}: {} nodes, {leaves} leaves, no problems",
        dir.display(),
        tree.nodes().len()
    );
    Ok(())
}

/// `~/jevons/logs/jevons-desktop.log`; the previous run's log is kept next to it.
fn log_file() -> Option<std::fs::File> {
    let dir = jevons_desktop_core::config::user_dir().join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let file = dir.join("jevons-desktop.log");
    let _ = std::fs::rename(&file, dir.join("jevons-desktop.previous.log"));
    std::fs::File::create(file).ok()
}

/// One headless take, from audio or text, for scripted end-to-end checks.
fn replay(
    args: &Args,
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
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    // Headless runs list the MCP servers' tools but never run one.
    let tools = Arc::new(
        jevons_desktop_core::flow::tools::ToolHost::new(&config.tools, &config.mcp).dry_run(),
    );
    for problem in tokio.block_on(tools.start()) {
        eprintln!("note: {problem}");
    }
    let (flows, notes) = defaults::open(&config.flows_dir(&config_file), &tools.catalog());
    for note in notes {
        eprintln!("note: {note}");
    }
    if !flows.is_valid() {
        for error in &flows.errors {
            eprintln!("{error}");
        }
        return Err("the flow tree has problems; see above".into());
    }
    let inspector: Arc<dyn jevons_desktop_core::platform::ContextInspector> = match &args.tree {
        Some(file) => Arc::new(jevons_desktop_core::recorded::RecordedInspector::load(
            file,
        )?),
        None => platform::context_inspector(),
    };
    let investigator = connection.models.generative.clone().map(|model| {
        Arc::new(jevons_desktop_core::flow::investigator::Investigator::new(
            connection.client.clone(),
            model,
            inspector,
            config.privacy.clone(),
            Arc::default(),
        )) as Arc<dyn jevons_desktop_core::flow::investigate::Investigate>
    });
    let dictation = &config.dictation;
    let env = Env {
        client: connection.client,
        flows: Arc::new(flows),
        settings: pipeline::Settings {
            models: connection.models,
            realtime: connection.realtime,
            language: dictation.language.clone(),
            decide: dictation.decide,
            max_output_tokens: dictation.max_output_tokens,
            ..pipeline::Settings::default()
        },
        sink: args
            .deliver
            .then(|| Arc::new(Mutex::new(platform::text_sink()))),
        investigator,
        // Headless runs never run a tool that asks first, and run no tool at all.
        confirmer: None,
        tools: Some(tools),
    };
    let trace = tokio.block_on(async {
        let (updates, mut live) = mpsc::unbounded_channel();
        let printer = tokio::spawn(async move {
            while let Some(update) = live.recv().await {
                match update {
                    pipeline::Update::Delta(text) | pipeline::Update::Output(text) => {
                        eprint!("{text}")
                    }
                    pipeline::Update::Step(step) => eprintln!("\n· {step}"),
                    _ => {}
                }
            }
            eprintln!();
        });
        let start = TakeStart {
            id: 1,
            context,
            entry: args.flow.clone(),
        };
        let trace = match (&args.replay, &args.transcript) {
            (Some(audio), _) => {
                let (events, received) = mpsc::unbounded_channel();
                let mut source = FileAudioSource {
                    file: audio.to_path_buf(),
                    paced: true,
                };
                let capture = source.start(None, events)?;
                let (_finish, finished) = oneshot::channel();
                let trace = pipeline::run_take(&env, start, received, finished, &updates).await;
                capture.stop();
                trace
            }
            (None, Some(text)) => pipeline::run_transcript(&env, start, text, &updates).await,
            (None, None) => unreachable!("replay runs with audio or a transcript"),
        };
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
                    inspector: platform::context_inspector(),
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
