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
#![forbid(unsafe_code)]
// A tray app: no console window on Windows. Logs go to a file; --replay output can be redirected.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod agent;
mod audio;
mod config;
mod hold;
mod platform;
mod runtime;
mod settings;
mod tray;
mod tuning;
mod ui;

use crate::config::{DesktopConfig, default_config_file};
use agent::{Agent, Command, Layers, View};
use clap::Parser;
use jevons_desktop_core::context::ContextSnapshot;
use jevons_desktop_core::fake::FileAudioSource;
use jevons_desktop_core::history::{self, History};
use jevons_desktop_core::platform::AudioSource;
use jevons_desktop_server::flow::{FlowTree, defaults};
use jevons_desktop_server::pipeline::{self, Env, TakeStart};
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
    /// Clear history and exit: `logs`, `traces` (of takes and automation runs), `trees`
    /// (recorded interfaces), `recordings` (demonstrations) or `all`; several separated by
    /// commas.
    #[arg(long, value_name = "WHAT", value_enum, value_delimiter = ',')]
    clear: Vec<Clear>,
}

/// What `--clear` clears.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Clear {
    Logs,
    Traces,
    Trees,
    Recordings,
    All,
}

impl Clear {
    fn kinds(self) -> &'static [History] {
        match self {
            Self::Logs => &[History::Logs],
            Self::Traces => &[History::Traces],
            Self::Trees => &[History::Trees],
            Self::Recordings => &[History::Recordings],
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
        return replay(&args, config, config_file);
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
    let desk = jevons_desktop_core::desk::LocalDesk::default().with_automations(automations);
    let catalog = jevons_desktop_server::flow::tools::ToolHost::new(&config.tools, &config.mcp)
        .with_desk(Arc::new(desk))
        .catalog();
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
    runtime.apply(&config, &config_file);
    let routes = loop {
        changed.recv()?;
        match runtime.status() {
            runtime::Status::Loading => continue,
            status => match runtime.routes() {
                Some(routes) => {
                    // A provider that does not answer leaves its capabilities out.
                    if matches!(status, runtime::Status::Failed(_)) {
                        eprintln!("note: {}", status.describe());
                    }
                    break routes;
                }
                None => return Err(status.describe().into()),
            },
        }
    };
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    // Headless runs list the MCP servers' tools and the automations, but never run one.
    let automations = Arc::new(jevons_desktop_core::automation::host::AutomationHost::new(
        &config.automations_dir(&config_file),
        config.automation.clone(),
        Arc::new(jevons_desktop_core::platform::Unsupported),
        Arc::new(jevons_desktop_core::platform::Unsupported),
    ));
    let scripts = jevons_desktop_core::desk::LocalDesk::default().with_automations(automations);
    let tools = Arc::new(
        jevons_desktop_server::flow::tools::ToolHost::new(&config.tools, &config.mcp)
            .with_desk(Arc::new(scripts))
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
    let inspector: Arc<dyn jevons_desktop_core::platform::ContextInspector> = match &args.tree {
        Some(file) => Arc::new(jevons_desktop_core::recorded::RecordedInspector::load(
            file,
        )?),
        None => platform::context_inspector(),
    };
    // The client's side: the interface is read within the privacy settings, the text is
    // typed only with --deliver, and there is no one to confirm a tool.
    let reader =
        jevons_desktop_core::reader::Reader::new(inspector.clone(), config.privacy.clone());
    let looks =
        jevons_desktop_core::look::Looks::new(inspector, config.privacy.clone(), Arc::default());
    let mut desk = jevons_desktop_core::desk::LocalDesk::default()
        .with_reader(Arc::new(reader))
        .with_looks(Arc::new(looks));
    if args.deliver {
        desk = desk.with_sink(Arc::new(Mutex::new(platform::text_sink())));
    }
    let desk: Arc<dyn jevons_desktop_protocol::desk::Desk> = Arc::new(desk);
    let investigator = routes.generation.as_ref().map(|route| {
        Arc::new(
            jevons_desktop_server::flow::investigator::Investigator::new(
                route.client.clone(),
                route.model.clone(),
                desk.clone(),
            ),
        ) as Arc<dyn jevons_desktop_server::flow::investigate::Investigate>
    });
    let dictation = &config.dictation;
    let env = Env {
        routes,
        flows: Arc::new(flows),
        settings: pipeline::Settings {
            language: dictation.language.clone(),
            decide: dictation.decide,
            max_output_tokens: dictation.max_output_tokens,
            ..pipeline::Settings::default()
        },
        desk,
        investigator,
        // Headless runs never run a tool that asks first, and run no tool at all.
        tools: Some(tools),
        machines: Arc::default(),
    };
    let traces = tokio.block_on(async {
        let (updates, mut live) = mpsc::unbounded_channel();
        let printer = tokio::spawn(async move {
            while let Some(update) = live.recv().await {
                match update {
                    pipeline::Update::Delta(text) | pipeline::Update::Output(text) => {
                        eprint!("{text}")
                    }
                    pipeline::Update::Stage(stage) => {
                        let choices = if stage.choices.is_empty() {
                            String::new()
                        } else {
                            format!(" ({})", stage.choices.join(", "))
                        };
                        eprintln!("\n· {:?} {}{choices}", stage.kind, stage.label);
                    }
                    pipeline::Update::StageDone { detail, chosen, ok } => {
                        let chosen = chosen.map_or(String::new(), |c| format!("{c} "));
                        eprintln!("  {} {chosen}{detail}", if ok { "✓" } else { "✗" });
                    }
                    pipeline::Update::State(path) => eprintln!("\n▸ {path}"),
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
                traces.push(pipeline::run_take(&env, start, received, finished, &updates).await);
                capture.stop();
            }
            None => {
                for (id, text) in (1..).zip(&args.transcript) {
                    let start = TakeStart {
                        id,
                        ..start.clone()
                    };
                    traces.push(pipeline::run_transcript(&env, start, text, &updates).await);
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
    runtime.shutdown();
    match traces.into_iter().find_map(|t| t.error) {
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
                    actor: platform::ui_actor(),
                    recorder: platform::recorder(),
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(Args::try_parse_from(["jevons-desktop", "--clear", "models"]).is_err());
        let reset = Args::try_parse_from(["jevons-desktop", "--reset-settings"]).unwrap();
        assert!(reset.headless());
    }
}
