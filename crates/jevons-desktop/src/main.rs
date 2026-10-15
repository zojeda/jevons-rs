//! jevons-desktop: a context-aware desktop agent in the tray.
//!
//! Tap the hotkey to toggle dictation or hold it while speaking. The agent reads the focused
//! application and field, transcribes, and walks the flow tree (a folder of TOML files) to a
//! leaf that types into the application, answers in the bubble or calls a tool.
//!
//! `--replay <audio>` or `--transcript <text>`, with `--context <json>`, runs one take headless
//! and prints its trace. `--check-flows` and `--init-flows` check and prepare a flows folder.
//! `--xpath <expr>` prints what an expression selects in the interface (or in `--tree`).
//! `--check-automations`, `--dry-run <name>` and `--run <name>` check, replay and run the
//! automations library. `--reset-settings` puts the default settings folder back (its git
//! history keeps the earlier one), and `--clear <what>` clears logs, traces and recordings.
//! `--say <text>` speaks through the voice route, or writes a WAV file with `--out`.
#![forbid(unsafe_code)]
// A tray app: no console window on Windows. Logs go to a file; --replay output can be redirected.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod agent;
mod audio;
mod config;
mod control;
mod hold;
mod platform;
mod runtime;
mod settings;
mod tray;
mod tuning;
mod ui;
mod voice;

use crate::config::{DesktopConfig, default_config_file};
use agent::{Agent, Command, Layers, View};
use clap::Parser;
use jevons_desktop_core::context::ContextSnapshot;
use jevons_desktop_core::fake::FileAudioSource;
use jevons_desktop_core::history::{self, History};
use jevons_desktop_core::platform::{AudioSource, MenuCommand, SAMPLE_RATE, Speaker};
use jevons_desktop_server::client::{Pcm16, Route, Routes, SpeechRequest};
use jevons_desktop_server::flow::{FlowTree, defaults};
use jevons_desktop_server::pipeline::{self, TakeStart};
use std::path::{Path, PathBuf};
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
    /// Run one take from this text as if it had been said, print its trace and exit. Repeat it
    /// for several takes in a row: they share the machines, so a task waits between them.
    #[arg(long, value_name = "TEXT")]
    transcript: Vec<String>,
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
    #[arg(long, value_name = "DIR")]
    #[allow(clippy::option_option)]
    check_flows: Option<Option<PathBuf>>,
    /// Write the built-in flow tree into a folder that has none (by default the settings'
    /// flows folder), refresh AGENTS.md and the schemas, and exit.
    #[arg(long, value_name = "DIR")]
    #[allow(clippy::option_option)]
    init_flows: Option<Option<PathBuf>>,
    /// Evaluate an XPath expression against the window in front (or --app's, or --tree's),
    /// print what it selects and exit.
    #[arg(long, value_name = "EXPR")]
    xpath: Option<String>,
    /// With --xpath: the window of the first application whose process name matches this glob.
    #[arg(long, value_name = "GLOB")]
    app: Option<String>,
    /// Check every automation of a library (by default the settings' one): the script checks,
    /// then each fixture's dry run. Prints the problems and exits.
    #[arg(long, value_name = "DIR")]
    #[allow(clippy::option_option)]
    check_automations: Option<Option<PathBuf>>,
    /// Replay an automation against a recorded demonstration (its first fixture, or --recording)
    /// and print the run's trace.
    #[arg(long, value_name = "NAME", conflicts_with = "run")]
    dry_run: Option<String>,
    /// Run an approved automation on the live interface and print the run's trace.
    #[arg(long, value_name = "NAME")]
    run: Option<String>,
    /// With --dry-run: the demonstration's JSON file.
    #[arg(long, value_name = "JSON")]
    recording: Option<PathBuf>,
    /// With --dry-run or --run: the arguments, as a JSON object.
    #[arg(long, value_name = "JSON")]
    args: Option<String>,
    /// With --dry-run, --run or --author: the automations library (by default the settings'
    /// one).
    #[arg(long, value_name = "DIR")]
    library: Option<PathBuf>,
    /// Write an automation into the library from a saved recording's folder, drafted from the
    /// recording alone, and print its checks.
    #[arg(long, value_name = "RECORDING")]
    author: Option<PathBuf>,
    /// With --author: replace this automation (a new version, to approve again).
    #[arg(long, value_name = "NAME")]
    replace: Option<String>,
    /// Put the default settings, flow tree and automations library back in the settings
    /// folder, and exit. Its git history keeps what was there. Quit the tray app first, or use
    /// its menu instead.
    #[arg(long)]
    reset_settings: bool,
    /// Serve desktop clients and nothing else: no tray and no window. It loads the settings,
    /// the models routed to it and the flows folder, and listens on the settings' address
    /// (`ws://<bind>:<port>/desktop`) until it is stopped.
    #[arg(long, conflicts_with_all = ["replay", "transcript", "server"])]
    serve: bool,
    /// With --replay or --transcript: run the take on the jevons server at this address, such
    /// as `ws://127.0.0.1:8080/desktop`, instead of in this process. This side still reads
    /// the screen and, with --deliver, types.
    #[arg(long, value_name = "URL")]
    server: Option<String>,
    /// The key the server at --server asks for; `TYPESAFE_API_KEY` when left out.
    #[arg(long, value_name = "KEY", requires = "server")]
    key: Option<String>,
    /// Clear history and exit: `logs`, `traces` (of takes and automation runs), `trees`
    /// (recorded interfaces), `recordings` (demonstrations), `machines` (the tasks kept across
    /// restarts) or `all`; several separated by commas.
    #[arg(long, value_name = "WHAT", value_enum, value_delimiter = ',')]
    clear: Vec<Clear>,
    /// Speak this text through the voice route, on the settings' output device, and exit.
    #[arg(long, value_name = "TEXT", conflicts_with_all = ["replay", "transcript", "serve", "server"])]
    say: Option<String>,
    /// With --say: write the speech to this WAV file in place of playing it.
    #[arg(long, value_name = "WAV", requires = "say")]
    out: Option<PathBuf>,
    /// Check that GPU `INDEX` lets this process in, and exit. The app runs a copy of itself
    /// with this before it first loads models, to wait for a GPU that does not answer yet.
    #[arg(long, value_name = "INDEX", hide = true)]
    probe_gpu: Option<usize>,
}

/// What `--clear` clears.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Clear {
    Logs,
    Traces,
    Trees,
    Recordings,
    Machines,
    All,
}

impl Clear {
    fn kinds(self) -> &'static [History] {
        match self {
            Self::Logs => &[History::Logs],
            Self::Traces => &[History::Traces],
            Self::Trees => &[History::Trees],
            Self::Recordings => &[History::Recordings],
            Self::Machines => &[History::Machines],
            Self::All => &History::ALL,
        }
    }
}

impl Args {
    fn headless(&self) -> bool {
        self.replay.is_some()
            || !self.transcript.is_empty()
            || self.check_flows.is_some()
            || self.init_flows.is_some()
            || self.xpath.is_some()
            || self.check_automations.is_some()
            || self.dry_run.is_some()
            || self.run.is_some()
            || self.author.is_some()
            || self.reset_settings
            || !self.clear.is_empty()
            || self.serve
            || self.say.is_some()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    // Before the log is opened: the app asks a copy of itself, which leaves the app's log alone.
    if let Some(index) = args.probe_gpu {
        return runtime::probe_gpu(index);
    }
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
    // Before loading the settings: a reset also mends a file that does not load.
    if args.reset_settings {
        return reset_settings(&config_file);
    }
    let config = DesktopConfig::load(&config_file)?;
    if !args.clear.is_empty() {
        return clear_history(&args.clear, &config);
    }
    let folder = |dir: &Option<PathBuf>| {
        dir.clone()
            .unwrap_or_else(|| config.flows_dir(&config_file))
    };
    if let Some(dir) = &args.check_flows {
        return check_flows(&folder(dir), &config, &config_file);
    }
    if let Some(dir) = &args.init_flows {
        let dir = folder(dir);
        let report = defaults::init(&dir)?;
        for file in &report.written {
            println!("wrote {}", dir.join(file).display());
        }
        for file in &report.removed {
            println!("removed {}", dir.join(file).display());
        }
        commit_written(
            &config_file,
            report.changed().map(|f| dir.join(f)).collect(),
            "Write the built-in flow tree (--init-flows)",
        );
        for note in &report.notes {
            println!("note: {note}");
        }
        return check_flows(&dir, &config, &config_file);
    }
    if let Some(expression) = &args.xpath {
        return xpath(&args, expression);
    }
    if let Some(dir) = &args.check_automations {
        let dir = dir
            .clone()
            .unwrap_or_else(|| config.automations_dir(&config_file));
        return check_automations(&dir, &config);
    }
    if args.dry_run.is_some() || args.run.is_some() {
        return run_automation(&args, &config, &config_file);
    }
    if let Some(recording) = &args.author {
        return author_automation(&args, recording, &config, &config_file);
    }
    if args.replay.is_some() || !args.transcript.is_empty() {
        return match &args.server {
            Some(url) => replay_on(&args, url, &config, &config_file),
            None => replay(&args, config, config_file),
        };
    }
    if args.serve {
        return serve(config, config_file);
    }
    if let Some(text) = &args.say {
        return say(text, args.out.as_deref(), &config, &config_file);
    }
    desktop(config, config_file)
}

/// Prints what an expression selects: one line per element, or the value. It reads only the
/// window in front (or `--app`'s), which is also the root's one window, so `//` stays in it.
fn xpath(args: &Args, expression: &str) -> Result<(), Box<dyn std::error::Error>> {
    use jevons_desktop_core::platform::ContextInspector;
    use jevons_desktop_core::xpath::{self, Document, Variables, XPath};
    let parsed = XPath::parse(expression).map_err(|e| {
        eprintln!(
            "{expression}\n{}^ {}",
            " ".repeat(e.column.saturating_sub(1)),
            e.message
        );
        format!("not an expression (column {})", e.column)
    })?;
    let inspector: Arc<dyn ContextInspector> = match &args.tree {
        Some(file) => Arc::new(jevons_desktop_core::recorded::RecordedInspector::load(
            file,
        )?),
        None => platform::context_inspector(),
    };
    let windows = inspector.windows()?;
    let chosen = match &args.app {
        Some(pattern) => {
            let glob = globset::GlobBuilder::new(pattern)
                .case_insensitive(true)
                .build()?
                .compile_matcher();
            windows.iter().position(|w| glob.is_match(&w.app))
        }
        None => windows.iter().position(|w| w.front).or(Some(0)),
    }
    .filter(|i| *i < windows.len())
    .ok_or("no window to read (is the application open?)")?;
    let windows = [windows[chosen].clone()];
    eprintln!("in {} {:?}", windows[0].app, windows[0].title);
    let began = std::time::Instant::now();
    let mut document = Document::new(&*inspector, &windows);
    let context = document.window(0).ok_or("no window to read")?;
    let value = document.evaluate(&parsed, context, &Variables::new())?;
    let lines = xpath::describe(&mut document, &value, 100);
    for line in &lines {
        println!("{line}");
    }
    eprintln!(
        "{} in {} ms, {} elements read",
        match &value {
            xpath::Value::Nodes(nodes) => format!("{} matches", nodes.len()),
            _ => "a value".into(),
        },
        began.elapsed().as_millis(),
        document.read()
    );
    Ok(())
}

/// Checks every automation in `dir`: prints each one's problems and fixtures, and fails when
/// one has a problem.
fn check_automations(
    dir: &std::path::Path,
    config: &DesktopConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use jevons_desktop_core::automation::{check, library::Library};
    let library = Library::load(dir);
    let mut problems = library.errors.len();
    for error in &library.errors {
        eprintln!("{}: {}", error.name, error.message);
    }
    for automation in library.automations.values() {
        let report = check::check(automation);
        let approved = automation.is_approved(&config.automation.approved);
        println!(
            "{}: {} ({})",
            automation.name,
            if report.ok() { "ok" } else { "problems" },
            if approved {
                "approved"
            } else {
                "not approved: approve it in the app to run it"
            }
        );
        for error in &report.errors {
            eprintln!("  {error}");
        }
        for fixture in &report.fixtures {
            match &fixture.problem {
                None => println!(
                    "  {}: replays {} steps",
                    fixture.recording,
                    fixture
                        .trace
                        .as_ref()
                        .and_then(|t| t.replayed)
                        .map_or(0, |(done, _)| done)
                ),
                Some(problem) => eprintln!("  {}: {problem}", fixture.recording),
            }
        }
        let summary = &report.summary;
        let mut does: Vec<String> = summary.actions.iter().cloned().collect();
        does.extend(summary.keys.iter().map(|k| format!("press {k}")));
        if summary.types_text {
            does.push("types into the window in front".into());
        }
        println!(
            "  in {}: {}",
            summary.apps.join(", "),
            if does.is_empty() {
                "reads only".to_string()
            } else {
                does.join(", ")
            }
        );
        if !report.ok() {
            problems += 1;
        }
    }
    if problems > 0 {
        return Err(format!("{}: {problems} automation(s) with problems", dir.display()).into());
    }
    println!(
        "{}: {} automation(s), no problems",
        dir.display(),
        library.automations.len()
    );
    Ok(())
}

/// `--author`: an automation from a saved recording, drafted without a model.
fn author_automation(
    args: &Args,
    recording_dir: &std::path::Path,
    config: &DesktopConfig,
    config_file: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use jevons_desktop_core::automation::author;
    use jevons_desktop_core::recording::bundle;
    let library = args
        .library
        .clone()
        .unwrap_or_else(|| config.automations_dir(config_file));
    let recording = bundle::load(recording_dir)?;
    let name = recording_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recording".into());
    let authored = tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(author::author(
            None,
            &recording,
            &name,
            &library,
            args.replace.as_deref(),
        ))?;
    commit_written(
        config_file,
        vec![authored.automation.dir.clone()],
        &format!(
            "Write the automation {} from a recording (--author)",
            authored.automation.name
        ),
    );
    let report = &authored.report;
    println!(
        "wrote {} ({}): {}",
        authored.automation.name,
        authored.automation.dir.display(),
        if report.ok() {
            "its checks pass; approve it in the app to run it"
        } else {
            "it has problems"
        }
    );
    for error in &report.errors {
        eprintln!("  {error}");
    }
    for fixture in &report.fixtures {
        if let Some(problem) = &fixture.problem {
            eprintln!("  {}: {problem}", fixture.recording);
        }
    }
    if report.ok() {
        Ok(())
    } else {
        Err("the automation does not pass its checks".into())
    }
}

/// `--dry-run` and `--run`: one automation, printing its trace as JSON.
fn run_automation(
    args: &Args,
    config: &DesktopConfig,
    config_file: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use jevons_desktop_core::automation::{host::AutomationHost, library::Library, run};
    let dir = args
        .library
        .clone()
        .unwrap_or_else(|| config.automations_dir(config_file));
    let given = match &args.args {
        Some(text) => Some(
            serde_json::from_str::<serde_json::Value>(text)?
                .as_object()
                .cloned()
                .ok_or("--args is a JSON object, such as '{\"channel\": \"random\"}'")?,
        ),
        None => None,
    };
    let trace = if let Some(name) = &args.dry_run {
        let library = Library::load(&dir);
        let automation = library
            .get(name)
            .ok_or_else(|| format!("{}: there is no automation {name:?}", dir.display()))?;
        let (demonstration, fixture_args) = match &args.recording {
            Some(file) => (
                serde_json::from_str(&std::fs::read_to_string(file)?)?,
                serde_json::Map::new(),
            ),
            None => automation
                .fixture(0)
                .map_err(|e| format!("{name}: {e} (give a demonstration with --recording)"))?,
        };
        run::dry_run(automation, demonstration, &given.unwrap_or(fixture_args))
    } else {
        let name = args.run.as_deref().expect("--run");
        let host = AutomationHost::new(
            &dir,
            config.automation.clone(),
            platform::context_inspector(),
            platform::ui_actor(),
        );
        host.run(
            name,
            &given.unwrap_or_default(),
            Some(Arc::new(|step: &str| eprintln!("· {step}"))),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false,
        )
    };
    println!("{}", serde_json::to_string_pretty(&trace)?);
    match &trace.error {
        None => Ok(()),
        Some(error) => Err(match (error.line, error.column) {
            (Some(line), Some(column)) => {
                format!("{} (script.rhai:{line}:{column})", error.message)
            }
            _ => error.message.clone(),
        }
        .into()),
    }
}

/// Prints every problem of the flow tree in `dir`; fails when there is one.
fn check_flows(
    dir: &std::path::Path,
    config: &DesktopConfig,
    config_file: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Tool names are checked against the settings and the automations library; MCP tools by
    // server (they are not started).
    let automations = Arc::new(jevons_desktop_core::automation::host::AutomationHost::new(
        &config.automations_dir(config_file),
        config.automation.clone(),
        Arc::new(jevons_desktop_core::platform::Unsupported),
        Arc::new(jevons_desktop_core::platform::Unsupported),
    ));
    let desk = jevons_desktop_core::desk::LocalDesk::default()
        .with_automations(automations)
        .with_tools(Arc::new(jevons_desktop_tools::ToolSet::new(
            &config.desk_tools,
            &config.desk_mcp,
        )));
    let tools = jevons_desktop_server::flow::tools::ToolHost::new(&config.tools, &config.mcp)
        .with_desk(Arc::new(desk));
    for clash in tools.clashes() {
        eprintln!("{clash}");
    }
    let catalog = tools.catalog();
    let tree = FlowTree::load(&jevons_desktop_server::flow::Disk::new(dir), &catalog);
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
    // Where the decision model may be asked, and where it never is.
    for (machine, decides) in tree.decided() {
        match &decides.event {
            Some(event) => println!("{machine} · {} on {event}: {}", decides.at, decides.by),
            None => println!("{machine} · {}: {}", decides.at, decides.by),
        }
    }
    Ok(())
}

/// `~/jevons/logs/jevons-desktop.log`; the previous run's log is kept next to it. It is opened
/// to append, so clearing the logs can empty it while the app writes.
fn log_file() -> Option<std::fs::File> {
    let dir = crate::config::user_dir().join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let file = dir.join(history::DESKTOP_LOG);
    let _ = std::fs::rename(&file, dir.join("jevons-desktop.previous.log"));
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)
        .ok()
}

/// `--reset-settings`: the defaults back in the settings folder.
fn reset_settings(config_file: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let report = crate::settings::reset(config_file).map_err(|e| e.to_string())?;
    if !report.removed.is_empty() {
        println!("removed {}", report.removed.join(", "));
    }
    println!(
        "{}: the default settings, flow tree and automations library{}",
        report.dir.display(),
        if report.repository.is_some() {
            ", committed"
        } else {
            ""
        }
    );
    for note in &report.notes {
        println!("note: {note}");
    }
    Ok(())
}

/// `--clear`: each kind of history asked for, once.
fn clear_history(what: &[Clear], config: &DesktopConfig) -> Result<(), Box<dyn std::error::Error>> {
    let kinds: std::collections::BTreeSet<History> =
        what.iter().flat_map(|w| w.kinds()).copied().collect();
    let mut failed = false;
    for kind in kinds {
        let cleared = history::clear(kind, &config.client());
        println!("{cleared} in {}", cleared.dir.display());
        failed |= !cleared.failed.is_empty();
    }
    if failed {
        return Err("some files could not be removed; see above".into());
    }
    Ok(())
}

/// Commits files a headless command wrote in the settings folder, when it is jevons' repository.
fn commit_written(config_file: &std::path::Path, paths: Vec<PathBuf>, message: &str) {
    let dir = crate::settings::folder(config_file);
    if let Some(repository) = jevons_desktop_core::git::Repository::open(dir)
        && let Err(e) = repository.commit(&paths, message)
    {
        eprintln!("note: cannot commit to the settings folder: {e}");
    }
}

/// One headless take, from audio or text, for scripted end-to-end checks.
fn replay(
    args: &Args,
    config: DesktopConfig,
    config_file: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::config::log_api(config.log_api);
    let context: ContextSnapshot = match &args.context {
        Some(file) => serde_json::from_str(&std::fs::read_to_string(file)?)?,
        None => ContextSnapshot::default(),
    };
    let (ready, changed) = std::sync::mpsc::channel();
    let runtime = runtime::Runtime::start(move || {
        let _ = ready.send(());
    });
    // However the run ends, the models unload before the process does.
    let _unload = runtime::ShutdownOnDrop(runtime.clone());
    runtime.apply(&config, &config_file);
    let routes = served_routes(&runtime, &changed)?;
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let desk = headless_desk(args, &config, &config_file)?;
    let tools = Arc::new(
        jevons_desktop_server::flow::tools::ToolHost::new(&config.tools, &config.mcp)
            .with_desk(desk.clone())
            .dry_run(),
    );
    for problem in tokio.block_on(tools.start()) {
        eprintln!("note: {problem}");
    }
    let (flows, report) = defaults::open(&config.flows_dir(&config_file), &tools.catalog());
    for note in report.notes {
        eprintln!("note: {note}");
    }
    if !flows.is_valid() {
        for error in &flows.errors {
            eprintln!("{error}");
        }
        return Err("the flow tree has problems; see above".into());
    }
    // A headless run has no speaker: a leaf that asks for voice answers as the bubble's does.
    let settings = config.take_settings(false);
    let flows = Arc::new(flows);
    let traces = tokio.block_on(async {
        // The server's side, in this process. Headless runs never run a tool that asks
        // first, and run no tool at all.
        let (session, _events) =
            jevons_desktop_server::session::Session::open(desk, flows, settings);
        session.set_routes(Some(routes));
        session.set_tools(Some(tools));
        let (updates, mut live) = mpsc::unbounded_channel();
        let printer = tokio::spawn(async move {
            while let Some(update) = live.recv().await {
                print_update(&update);
            }
            eprintln!();
        });
        let start = TakeStart {
            id: 1,
            context,
            entry: args.flow.clone(),
        };
        let mut traces = Vec::new();
        match &args.replay {
            Some(audio) => {
                let (events, received) = mpsc::unbounded_channel();
                let mut source = FileAudioSource {
                    file: audio.to_path_buf(),
                    paced: true,
                };
                let capture = source.start(None, events)?;
                let (_finish, finished) = oneshot::channel();
                traces.push(
                    session
                        .take(start, received, finished, &updates, None)
                        .await,
                );
                capture.stop();
            }
            None => {
                for (id, text) in (1..).zip(&args.transcript) {
                    let start = TakeStart {
                        id,
                        ..start.clone()
                    };
                    traces.push(session.transcript(start, text, &updates).await);
                }
            }
        }
        drop(updates);
        let _ = printer.await;
        Ok::<_, Box<dyn std::error::Error>>(traces)
    })?;
    match traces.as_slice() {
        [trace] => println!("{}", serde_json::to_string_pretty(trace)?),
        all => println!("{}", serde_json::to_string_pretty(all)?),
    }
    match traces.into_iter().find_map(|t| t.error) {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}

/// Waits for the runtime to serve what the settings route, and returns the routes. With
/// nothing served it fails with the runtime's status; a provider that does not answer is a
/// note, and its capabilities are left out.
fn served_routes(
    runtime: &runtime::Runtime,
    changed: &std::sync::mpsc::Receiver<()>,
) -> Result<Routes, Box<dyn std::error::Error>> {
    loop {
        changed.recv()?;
        match runtime.status() {
            runtime::Status::Loading => continue,
            status @ runtime::Status::WaitingForGpu { .. } => {
                eprintln!("note: {}", status.describe());
                continue;
            }
            status => match runtime.routes() {
                Some(routes) => {
                    if matches!(status, runtime::Status::Failed(_)) {
                        eprintln!("note: {}", status.describe());
                    }
                    return Ok(routes);
                }
                None => return Err(status.describe().into()),
            },
        }
    }
}

/// `--say`: the text through the voice route, played on the settings' output device, or
/// written as a WAV file with `--out`. Only the voice is loaded and asked: it speaks whatever
/// the "Speak answers" switch says, since the command asks for it.
fn say(
    text: &str,
    out: Option<&Path>,
    config: &DesktopConfig,
    config_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (ready, changed) = std::sync::mpsc::channel();
    let runtime = runtime::Runtime::start(move || {
        let _ = ready.send(());
    });
    // However the run ends, the model unloads before the process does.
    let _unload = runtime::ShutdownOnDrop(runtime.clone());
    runtime.only(crate::config::Capability::Voice);
    runtime.apply(config, config_file);
    let nothing = "nothing serves the voice: choose a voice model in the Models tab, or route \
                   `voice` to a provider in jevons-server.toml";
    let routes = match served_routes(&runtime, &changed) {
        Ok(routes) => routes,
        // No model at all is this command's own case to explain; a voice model that did not
        // load, or a provider that does not answer, says what is wrong itself.
        Err(_) if runtime.status() == runtime::Status::NoModels => return Err(nothing.into()),
        Err(e) => return Err(e),
    };
    let route = routes.voice.ok_or(nothing)?;
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let began = std::time::Instant::now();
    let mut first: Option<u128> = None;
    let samples = match out {
        Some(file) => {
            let mut all = Vec::new();
            tokio.block_on(speech(&route, text, |stretch| {
                first.get_or_insert_with(|| began.elapsed().as_millis());
                all.extend_from_slice(stretch);
                true
            }))?;
            std::fs::write(file, wav(&all)).map_err(|e| format!("{}: {e}", file.display()))?;
            println!("wrote {}", file.display());
            all.len()
        }
        None => {
            let mut playback = audio::CpalSpeaker.start(config.voice.output.as_deref())?;
            let samples = tokio.block_on(speech(&route, text, |stretch| {
                first.get_or_insert_with(|| began.elapsed().as_millis());
                playback.play(stretch)
            }))?;
            // All of it is queued long before the last word is heard.
            while playback.playing() {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            samples
        }
    };
    println!(
        "{:.1} s of speech from {}; the first audio after {} ms, all of it after {} ms",
        samples as f64 / f64::from(SAMPLE_RATE),
        route.name(),
        first.unwrap_or_default(),
        began.elapsed().as_millis()
    );
    Ok(())
}

/// Asks `route` for `text` as speech and hands the samples to `heard` as they arrive, mono
/// PCM16 at [`SAMPLE_RATE`]. Returns how many came. It ends early when `heard` answers false
/// or the provider gives nothing for as long as a take would wait.
async fn speech(
    route: &Route,
    text: &str,
    mut heard: impl FnMut(&[i16]) -> bool,
) -> Result<usize, String> {
    let limit = pipeline::Settings::default().voice_timeout;
    let late = || {
        format!(
            "no speech from {} within {} s",
            route.name(),
            limit.as_secs()
        )
    };
    let failed = |e| format!("{} failed: {e}", route.name());
    let request = SpeechRequest {
        model: route.model.clone(),
        input: text.to_string(),
        voice: route.voice.clone(),
        language: None,
    };
    let mut stream = tokio::time::timeout(limit, route.client.speak(&request))
        .await
        .map_err(|_| late())?
        .map_err(failed)?;
    let mut pcm = Pcm16::default();
    let mut samples = Vec::new();
    let mut count = 0;
    loop {
        let piece = tokio::time::timeout(limit, stream.next())
            .await
            .map_err(|_| late())?
            .map_err(failed)?;
        let Some(bytes) = piece else { break };
        samples.clear();
        pcm.push(&bytes, &mut samples);
        count += samples.len();
        if !samples.is_empty() && !heard(&samples) {
            return Err("the output device stopped playing".into());
        }
    }
    if count == 0 {
        return Err(format!("{} answered with no audio", route.name()));
    }
    Ok(count)
}

/// A WAV file of mono PCM16 at [`SAMPLE_RATE`].
fn wav(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend(jevons_audio::wav_header(
        SAMPLE_RATE,
        u32::try_from(samples.len()).ok(),
    ));
    bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
    bytes
}

/// The client's side of a headless take: the interface is read within the privacy settings
/// (from `--tree` when given), the text is typed only with --deliver, there is no one to
/// confirm a tool, and no tool runs. The library's automations and the client's own tools are
/// listed all the same, so flows that name them load.
fn headless_desk(
    args: &Args,
    config: &DesktopConfig,
    config_file: &Path,
) -> Result<Arc<dyn jevons_desktop_protocol::desk::Desk>, Box<dyn std::error::Error>> {
    let automations = Arc::new(jevons_desktop_core::automation::host::AutomationHost::new(
        &config.automations_dir(config_file),
        config.automation.clone(),
        Arc::new(jevons_desktop_core::platform::Unsupported),
        Arc::new(jevons_desktop_core::platform::Unsupported),
    ));
    let inspector: Arc<dyn jevons_desktop_core::platform::ContextInspector> = match &args.tree {
        Some(file) => Arc::new(jevons_desktop_core::recorded::RecordedInspector::load(
            file,
        )?),
        None => platform::context_inspector(),
    };
    let reader =
        jevons_desktop_core::reader::Reader::new(inspector.clone(), config.privacy.clone());
    // Outside agents read here within the client's `[control]`; they act only with --deliver,
    // and there is no bubble to ask the user in.
    let actor: Arc<dyn jevons_desktop_core::platform::UiActor> = if args.deliver {
        platform::ui_actor()
    } else {
        Arc::new(jevons_desktop_core::platform::Unsupported)
    };
    let control = jevons_desktop_core::control::Control::new(
        inspector.clone(),
        actor,
        config.desk_control.clone(),
    );
    let looks =
        jevons_desktop_core::look::Looks::new(inspector, config.privacy.clone(), Arc::default());
    let mut desk = jevons_desktop_core::desk::LocalDesk::default()
        .with_reader(Arc::new(reader))
        .with_looks(Arc::new(looks))
        .with_control(Arc::new(control))
        .with_automations(automations)
        .with_tools(Arc::new(jevons_desktop_tools::ToolSet::new(
            &config.desk_tools,
            &config.desk_mcp,
        )))
        .dry_run();
    if args.deliver {
        desk = desk.with_sink(Arc::new(Mutex::new(platform::text_sink())));
    }
    Ok(Arc::new(desk))
}

/// Says on stderr what a headless take is doing.
fn print_update(update: &pipeline::Update) {
    match update {
        pipeline::Update::Delta(text) | pipeline::Update::Output(text) => eprint!("{text}"),
        pipeline::Update::Stage(stage) => {
            let choices = if stage.choices.is_empty() {
                String::new()
            } else {
                format!(" ({})", stage.choices.join(", "))
            };
            eprintln!("\n· {:?} {}{choices}", stage.kind, stage.label);
        }
        pipeline::Update::StageDone { detail, chosen, ok } => {
            let chosen = chosen.as_ref().map_or(String::new(), |c| format!("{c} "));
            eprintln!("  {} {chosen}{detail}", if *ok { "✓" } else { "✗" });
        }
        pipeline::Update::State(path) => eprintln!("\n▸ {path}"),
        _ => {}
    }
}

/// Runs headless takes on the server at `url`: this side sends each take and attends to what
/// the server asks of the desk, and prints the traces the server sends back.
fn replay_on(
    args: &Args,
    url: &str,
    config: &DesktopConfig,
    config_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use jevons_desktop_protocol::delivery::AudioEvent;
    use jevons_desktop_protocol::take::TakeSettings;
    use jevons_desktop_protocol::wire::{Said, ToClient, ToServer, VERSION, attend};
    let context: ContextSnapshot = match &args.context {
        Some(file) => serde_json::from_str(&std::fs::read_to_string(file)?)?,
        None => ContextSnapshot::default(),
    };
    // The client's tools are listed for the server's flows, and none runs.
    let desk = headless_desk(args, config, config_file)?;
    let key = args
        .key
        .clone()
        .or_else(|| std::env::var("TYPESAFE_API_KEY").ok());
    let dictation = &config.dictation;
    let settings = TakeSettings {
        language: dictation.language.clone(),
        decide: dictation.decide,
        max_output_tokens: dictation.max_output_tokens,
        // A headless run has no speaker.
        voice: false,
    };
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let traces = tokio.block_on(async {
        let link = jevons_desktop_protocol::socket::connect(url).await?;
        let tools = desk.tools();
        let (say, mut hear) = attend(desk, link);
        say.send(ToServer::Hello {
            version: VERSION,
            key,
            tools,
            settings,
        })?;
        match hear.recv().await {
            Some(ToClient::Welcome { .. }) => {}
            Some(ToClient::Closed { why }) => return Err(why.into()),
            _ => return Err(format!("{url} did not answer as a jevons server").into()),
        }
        // Each take's trace, once the server sends it; what it says on the way is printed.
        let mut traces: Vec<serde_json::Value> = Vec::new();
        let mut takes = Vec::new();
        match &args.replay {
            Some(audio) => {
                say.send(ToServer::Take {
                    take: 1,
                    context,
                    entry: args.flow.clone(),
                    said: Said::Take,
                })?;
                let (events, mut captured) = mpsc::unbounded_channel();
                let mut source = FileAudioSource {
                    file: audio.to_path_buf(),
                    paced: true,
                };
                let capture = source.start(None, events)?;
                let sending = say.clone();
                tokio::spawn(async move {
                    while let Some(event) = captured.recv().await {
                        // The meter is this side's; the server needs the sound.
                        if matches!(event, AudioEvent::Level(_)) {
                            continue;
                        }
                        let _ = sending.send(ToServer::Audio { take: 1, event });
                    }
                    capture.stop();
                });
                takes.push(1);
            }
            None => {
                for (take, text) in (1..).zip(&args.transcript) {
                    takes.push(take);
                    // One after the other: a task waits between them.
                    say.send(ToServer::Transcript {
                        take,
                        context: context.clone(),
                        entry: args.flow.clone(),
                        text: text.clone(),
                    })?;
                    traces.push(trace_of(&mut hear, take).await?);
                }
            }
        }
        for take in takes.into_iter().skip(traces.len()) {
            traces.push(trace_of(&mut hear, take).await?);
        }
        eprintln!();
        Ok::<_, Box<dyn std::error::Error>>(traces)
    })?;
    match traces.as_slice() {
        [trace] => println!("{}", serde_json::to_string_pretty(trace)?),
        all => println!("{}", serde_json::to_string_pretty(all)?),
    }
    match traces.iter().find_map(|t| t["error"].as_str()) {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}

/// Waits for take `take`'s trace from the server, printing what the take does meanwhile.
async fn trace_of(
    hear: &mut mpsc::UnboundedReceiver<jevons_desktop_protocol::wire::ToClient>,
    take: u64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use jevons_desktop_protocol::wire::ToClient;
    loop {
        match hear.recv().await {
            Some(ToClient::Update { update, .. }) => print_update(&update),
            Some(ToClient::Trace { take: of, trace }) if of == take => return Ok(trace),
            Some(ToClient::Closed { why }) => return Err(why.into()),
            Some(_) => {}
            None => return Err("the server closed the connection".into()),
        }
    }
}

/// The server alone: no tray and no window. It serves desktop clients on the settings'
/// address until the process is stopped.
fn serve(config: DesktopConfig, config_file: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use jevons_desktop_protocol::desk::Nobody;
    use jevons_desktop_server::flow::tools::ToolHost;
    use jevons_desktop_server::serve::Host;
    use jevons_desktop_server::session::Session;
    crate::config::log_api(config.log_api);
    // The defaults the folder lacks: the flows folder is the server's to read.
    for note in settings::prepare(&config_file, &config).notes {
        eprintln!("note: {note}");
    }
    let (ready, changed) = std::sync::mpsc::channel();
    let runtime = runtime::Runtime::start(move || {
        let _ = ready.send(());
    });
    // However the server ends, the models unload before the process does.
    let _unload = runtime::ShutdownOnDrop(runtime.clone());
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    // Ctrl+C ends the server instead of killing it with the models loaded.
    let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop = interrupted.clone();
    tokio.spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    let met = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let session = tokio.block_on(async {
        // Nobody is at the desk until a client connects.
        let builtin = FlowTree::load(
            &defaults::builtin(),
            &jevons_desktop_server::flow::Catalog::default(),
        );
        let (session, events) = Session::open(
            Arc::new(Nobody),
            Arc::new(builtin),
            pipeline::Settings::default(),
        );
        let tools = Arc::new(ToolHost::new(&config.tools, &config.mcp).with_desk(session.desk()));
        for problem in tools.start().await {
            eprintln!("note: {problem}");
        }
        // The flow tree is checked against the tools known now. The client's are known once
        // one says hello, and the tree is read again then: until then a flow that names one
        // of them does not load, and the built-in tree runs.
        let dir = config.flows_dir(&config_file);
        let load = {
            let (session, tools) = (session.clone(), tools.clone());
            move || {
                // The automations are the client's: with none connected, `script:` tools are
                // taken as there.
                let mut catalog = tools.catalog();
                catalog.servers.entry("script".into()).or_insert(false);
                let (flows, report) = defaults::open(&dir, &catalog);
                for note in report.notes {
                    eprintln!("note: {note}");
                }
                if flows.is_valid() {
                    session.set_flows(Arc::new(flows));
                } else {
                    for error in &flows.errors {
                        eprintln!("{error}");
                    }
                    eprintln!(
                        "note: the flow tree has problems: the tree before it runs until it is fixed"
                    );
                }
            }
        };
        load();
        session.set_tools(Some(tools));
        // The tasks that run are kept in the data folder. They come back once a client has
        // said hello (the host does it: the flow tree is checked against the client's tools
        // by then) and a provider answers.
        session.keep_machines_in(history::machines_file());
        let host = Host::new(session.clone(), events, config.exposed_key());
        let greeted = met.clone();
        host.when_greeted(Arc::new(move || {
            load();
            greeted.store(true, std::sync::atomic::Ordering::Relaxed);
        }));
        runtime.serve(host);
        runtime.control(session.clone());
        session
    });
    runtime.apply(&config, &config_file);
    let address = std::net::SocketAddr::new(config.server.bind, config.server.port);
    eprintln!("Serving desktop clients on ws://{address}/desktop");
    if config.control.enabled {
        match config.exposed_key() {
            Some(_) => eprintln!("Serving outside agents on http://{address}/mcp and /api/v1"),
            None => eprintln!("note: {}", jevons_desktop_server::control::NO_KEY),
        }
    }
    // The routes follow the providers for as long as the server runs.
    loop {
        if interrupted.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!("Unloading the models…");
            break;
        }
        match changed.recv_timeout(std::time::Duration::from_millis(250)) {
            Ok(()) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            // The runtime thread is gone: nothing more will be served.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        let status = runtime.status();
        eprintln!("{}", status.describe());
        session.set_routes(runtime.routes());
        // A client came before a provider answered: its tasks come back now.
        if session.ready() && met.load(std::sync::atomic::Ordering::Relaxed) {
            for note in tokio.block_on(session.restore()) {
                eprintln!("note: {note}");
            }
        }
    }
    Ok(())
}

/// Ends the app as Quit does when the window's thread ends, however it ends: the agent cancels
/// the take and removes the tray icon, and the models unload before the process does. A panic on
/// that thread, such as a renderer's, would otherwise end the process with the models loaded,
/// which to the GPU driver is a process killed in the middle of a call.
struct QuitOnDrop {
    commands: mpsc::UnboundedSender<Command>,
    runtime: runtime::Runtime,
}

impl Drop for QuitOnDrop {
    fn drop(&mut self) {
        // After Quit the agent is gone and this reaches nobody.
        let _ = self.commands.send(Command::Menu(MenuCommand::Quit));
        self.runtime.shutdown();
    }
}

/// The tray, the agent and the inspector window.
fn desktop(config: DesktopConfig, config_file: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let view = Arc::new(Mutex::new(View::default()));
    let (commands, received) = mpsc::unbounded_channel::<Command>();
    let runtime_commands = commands.clone();
    let runtime = runtime::Runtime::start(move || {
        let _ = runtime_commands.send(Command::RuntimeChanged);
    });
    let _quit = QuitOnDrop {
        commands: commands.clone(),
        runtime: runtime.clone(),
    };
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
            runtime,
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
    runtime: runtime::Runtime,
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
                    actor: platform::ui_actor(),
                    recorder: platform::recorder(),
                    sink: platform::text_sink(),
                    audio: Box::new(audio::CpalSource),
                    speaker: Box::new(audio::CpalSpeaker),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_on_the_windows_thread_quits_the_agent_and_unloads_the_models() {
        let (commands, mut received) = mpsc::unbounded_channel::<Command>();
        let runtime = runtime::Runtime::start(|| {});
        let quit = QuitOnDrop {
            commands,
            runtime: runtime.clone(),
        };
        // As the bubble's renderer did, on the thread that runs the windows.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _quit = quit;
            panic!("failed to get surface texture: Other");
        }));
        assert!(panicked.is_err());
        assert!(matches!(
            received.try_recv(),
            Ok(Command::Menu(MenuCommand::Quit))
        ));
        assert!(
            runtime.has_ended(),
            "the runtime thread still runs after the panic"
        );
    }

    /// A provider that answers each request for speech with `samples` as raw PCM, and keeps
    /// what it was asked. A model called `none` gets an error.
    fn provider(samples: Vec<i16>) -> (Route, Arc<Mutex<Vec<serde_json::Value>>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let kept = asked.clone();
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let Ok(mut socket) = socket else { break };
                // The whole request: its head, then as many bytes as the head announces.
                let mut request = Vec::new();
                let mut buffer = [0u8; 4096];
                let body = loop {
                    let read = socket.read(&mut buffer).unwrap();
                    request.extend_from_slice(&buffer[..read]);
                    let Some(head) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let length = String::from_utf8_lossy(&request[..head])
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= head + 4 + length {
                        break request[head + 4..].to_vec();
                    }
                };
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let refused = body["model"] == "none";
                kept.lock().unwrap().push(body);
                let (status, kind, bytes): (_, _, Vec<u8>) = if refused {
                    (
                        "404 Not Found",
                        "application/json",
                        br#"{"error": {"message": "The model `none` does not exist"}}"#.to_vec(),
                    )
                } else {
                    let pcm = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                    ("200 OK", "audio/pcm", pcm)
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: {kind}\r\ncontent-length: {}\r\n\
                     connection: close\r\n\r\n",
                    bytes.len()
                );
                let _ = socket.write_all(head.as_bytes());
                let _ = socket.write_all(&bytes);
            }
        });
        let client = jevons_desktop_server::client::Client::new(&base, None);
        let route = Route::ours(client, "box", "qwen3-tts").in_voice(Some("lucia".into()));
        (route, asked)
    }

    #[test]
    fn say_speaks_text_and_out_names_a_wav_file_for_it() {
        let say = Args::try_parse_from(["jevons-desktop", "--say", "Son las cinco."]).unwrap();
        assert_eq!(say.say.as_deref(), Some("Son las cinco."));
        assert!(say.out.is_none() && say.headless());
        let out = ["jevons-desktop", "--say", "Hola", "--out", "hola.wav"];
        let file = Args::try_parse_from(out).unwrap();
        assert_eq!(file.out, Some(PathBuf::from("hola.wav")));
        // A file needs something to say, and a take is another command.
        assert!(Args::try_parse_from(["jevons-desktop", "--out", "hola.wav"]).is_err());
        let both = ["jevons-desktop", "--say", "Hola", "--transcript", "hola"];
        assert!(Args::try_parse_from(both).is_err());
        assert!(Args::try_parse_from(["jevons-desktop", "--say", "Hola", "--serve"]).is_err());
    }

    #[tokio::test]
    async fn said_text_goes_to_the_voice_route_and_comes_back_as_a_wav_file() {
        let said: Vec<i16> = (0..2_400).map(|n| (n % 200 - 100) * 50).collect();
        let (route, asked) = provider(said.clone());
        let mut heard = Vec::new();
        let count = speech(&route, "Son las cinco.", |stretch| {
            heard.extend_from_slice(stretch);
            true
        })
        .await
        .unwrap();
        assert_eq!((count, &heard), (2_400, &said));
        // What was asked: the route's model in its voice, the text, as raw PCM.
        assert_eq!(
            asked.lock().unwrap()[0],
            serde_json::json!({"model": "qwen3-tts", "input": "Son las cinco.", "voice": "lucia",
                               "response_format": "pcm"})
        );
        // The file: a 44-byte header for mono PCM16 at 24 kHz with its true sizes, then the
        // samples, which decode back as they were.
        let file = wav(&heard);
        assert_eq!(file.len(), 44 + 4_800);
        assert_eq!(
            (&file[0..4], &file[8..16], &file[36..40]),
            (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
        );
        let field = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().unwrap());
        assert_eq!(
            (field(4), field(24), field(40)),
            (36 + 4_800, 24_000, 4_800)
        );
        let limits = jevons_audio::AudioLimits {
            max_bytes: 1 << 20,
            max_seconds: 10.0,
        };
        let decoded = jevons_audio::decode_audio(file, Some("wav"), SAMPLE_RATE, limits).unwrap();
        assert_eq!(jevons_audio::to_pcm16(&decoded), said);
        // A speaker that stops playing ends the command, and so does a provider that refuses.
        let stopped = speech(&route, "Hola.", |_| false).await;
        assert_eq!(stopped, Err("the output device stopped playing".into()));
        let none = Route::ours(route.client.clone(), "box", "none");
        let refused = speech(&none, "Hola.", |_| true).await.unwrap_err();
        assert!(
            refused.starts_with("box/none failed:") && refused.contains("does not exist"),
            "{refused}"
        );
    }

    #[tokio::test]
    async fn said_text_plays_through_a_speaker_until_it_was_heard() {
        use jevons_desktop_core::fake::RecordingSpeaker;
        let (route, _) = provider(vec![7; 1_200]);
        let mut speaker = RecordingSpeaker::new();
        let mut playback = speaker.start(Some("Headphones")).unwrap();
        let count = speech(&route, "Hola.", |stretch| playback.play(stretch))
            .await
            .unwrap();
        assert_eq!(count, 1_200);
        // Queued, and still to be heard: the command waits for it.
        assert!(playback.playing());
        speaker.hear();
        assert!(!playback.playing());
        let played = speaker.played();
        assert_eq!(played[0].device.as_deref(), Some("Headphones"));
        assert_eq!(played[0].samples, vec![7; 1_200]);
    }

    #[test]
    fn flow_commands_take_an_optional_folder() {
        let bare = Args::try_parse_from(["jevons-desktop", "--init-flows"]).unwrap();
        assert_eq!(bare.init_flows, Some(None));
        let dir = Args::try_parse_from(["jevons-desktop", "--check-flows", "D:/flows"]).unwrap();
        assert_eq!(dir.check_flows, Some(Some(PathBuf::from("D:/flows"))));
        let neither = Args::try_parse_from(["jevons-desktop"]).unwrap();
        assert!(!neither.headless());
    }

    #[test]
    fn clearing_takes_kinds_of_history_or_all_of_it() {
        let some = Args::try_parse_from(["jevons-desktop", "--clear", "logs,traces"]).unwrap();
        assert_eq!(some.clear, [Clear::Logs, Clear::Traces]);
        assert!(some.headless());
        let all = Args::try_parse_from(["jevons-desktop", "--clear", "all"]).unwrap();
        assert_eq!(all.clear[0].kinds(), History::ALL);
        let tasks = Args::try_parse_from(["jevons-desktop", "--clear", "machines"]).unwrap();
        assert_eq!(tasks.clear[0].kinds(), [History::Machines]);
        assert!(Args::try_parse_from(["jevons-desktop", "--clear", "models"]).is_err());
        let reset = Args::try_parse_from(["jevons-desktop", "--reset-settings"]).unwrap();
        assert!(reset.headless());
    }
}
