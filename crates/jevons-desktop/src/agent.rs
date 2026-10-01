//! The agent: hotkeys, context capture, microphone takes and the pipeline, with the
//! tray and the inspector as views. It runs on its own thread; takes run on the runtime's
//! workers so a slow generation never blocks the hotkey.

use crate::runtime::{Runtime, Status};
use crate::tray::Tray;
use jevons_desktop_core::automation::author::{self, Authored};
use jevons_desktop_core::automation::check::CheckReport;
use jevons_desktop_core::automation::host::AutomationHost;
use jevons_desktop_core::automation::run::RunTrace;
use jevons_desktop_core::config::{DesktopConfig, HotkeyMode};
use jevons_desktop_core::context::ContextSnapshot;
use jevons_desktop_core::flow::confirm::{ChannelConfirmer, Confirmation};
use jevons_desktop_core::flow::extract::{self, Reader};
use jevons_desktop_core::flow::investigate::Investigate;
use jevons_desktop_core::flow::investigator::{Investigator, PathCache};
use jevons_desktop_core::flow::tools::ToolHost;
use jevons_desktop_core::flow::walk::{self, FlowStep};
use jevons_desktop_core::flow::{Catalog, FlowError, FlowTree, defaults};
use jevons_desktop_core::icons::TrayState;
use jevons_desktop_core::interface;
use jevons_desktop_core::pipeline::{self, Env, StageKind, TakeStart, Trace, Update};
use jevons_desktop_core::platform::{
    AudioDevice, AudioSource, Binding, CaptureHandle, ContextInspector, ContextProvider,
    DeliveryOutcome, HotkeyAction, HotkeyEvent, MenuCommand, MenuModel, Recorder, RecordingHandle,
    TextSink, TrayBackend, UiActor, UiElement, WindowEntry,
};
use jevons_desktop_core::recorded::RecordedTree;
use jevons_desktop_core::recording::{Session, bundle};
use jevons_desktop_core::xpath::selector::Candidate;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

/// How many takes the inspector keeps.
const HISTORY: usize = 50;

/// Requests to the agent, from the tray, hotkeys, the inspector and finished takes.
#[derive(Debug)]
pub enum Command {
    Hotkey(HotkeyEvent),
    /// The hotkeys the tray registered, and those it could not.
    HotkeysRegistered {
        actions: Vec<(u32, HotkeyAction, String)>,
        errors: Vec<String>,
    },
    Menu(MenuCommand),
    /// Save and apply new settings.
    Apply(Box<DesktopConfig>),
    /// Read the context now, for the inspector.
    RefreshContext,
    /// Read the context after a delay, so the user can switch to the target application.
    CaptureContextIn(Duration),
    /// Saves the interface of the window in front, for writing flows and investigation tests.
    RecordTree,
    /// The automations library changed on disk.
    ReloadAutomations,
    /// What the user said while recording, transcribed.
    RecordingNote(Result<String, String>),
    /// A recording was saved (its folder, steps and description), or why not.
    RecordingSaved(Result<(PathBuf, usize, String), String>),
    /// The author wrote an automation from a recording, or why not.
    Authored(Result<Box<Authored>, String>),
    /// An automation's checks, for approving it.
    Reviewed(Box<CheckReport>),
    /// The user's answer to approving an automation's version.
    ApprovalAnswered {
        name: String,
        version: String,
        yes: bool,
    },
    /// What a running automation does now.
    AutomationProgress(String),
    AutomationFinished(Box<RunTrace>),
    /// Tries an extract on the window the Context tab shows: at once, or after a pause in typing.
    TryExtract(Box<TrialRequest>),
    /// Live mode: try the latest extract again whenever the Context tab's window changes.
    LiveTrials(bool),
    /// The pause after typing ended for the request with this number.
    TrialDue(u64),
    /// A trial finished.
    TrialDone(Box<TrialView>),
    /// Writes an edited extract into its node file.
    SaveExtract(Box<TrialRequest>),
    /// Reads the flow tree's extracts in the window the Context tab shows again.
    ReadExtracts,
    /// Reads the top of the interface of the window the Context tab shows, anew.
    InterfaceLoad,
    /// Opens an element of the interface: its children, or (`all`) everything below it.
    InterfaceOpen {
        id: String,
        all: bool,
    },
    /// Closes an element of the interface (its children stay read).
    InterfaceClose(String),
    /// Reads the next page of an element's children.
    InterfaceMore(String),
    /// Opens the interface from the window down to this many levels.
    InterfaceExpand(usize),
    /// Closes every element of the interface.
    InterfaceCollapse,
    /// Searches the whole window for elements holding this text (empty clears the search).
    InterfaceSearch(String),
    /// A search finished: its tree's generation, its text, and what it found or why not.
    InterfaceSearched(u64, String, Result<interface::Found, String>),
    /// Opens the interface down to an element and selects it: the ids from the window's child
    /// down, or just the element's (its ancestors are then found by walking up from it).
    InterfaceReveal(Vec<String>),
    /// Opens the interface down to the element that had the focus in the Context tab's snapshot.
    InterfaceFocus,
    /// Selects an element of the interface: its properties and the selectors that find it.
    InterfaceSelect(Box<UiElement>),
    /// The selectors of the selected element were found (or not).
    InterfaceFound,
    ReloadFlows,
    RuntimeChanged,
    /// The MCP servers listed their tools: check the flows against them.
    ToolsListed(Vec<String>),
    /// Apply the settings to the runtime again, such as after a model finished downloading.
    ReloadRuntime,
    TakeFinished(Box<Trace>),
    /// Hides the feedback bubble of a finished take, unless a newer take shows it.
    HideFeedback(u64),
    /// Hides a message, unless a newer one replaced it.
    HideMessage(u64),
    /// A take asks before calling a tool.
    ConfirmRequested(Confirmation),
    /// The user's answer: run the tool or not.
    Confirmed(bool),
    /// A button of the bubble's answer.
    Bubble(BubbleAction),
}

/// What the buttons under an answer in the bubble do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BubbleAction {
    Copy,
    /// Types the answer into the window the take started in.
    Insert,
    Close,
}

/// A tool call waiting for the user in the bubble.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PendingCall {
    pub tool: String,
    /// The arguments, as readable JSON.
    pub arguments: String,
}

/// A stage of the take as the bubble shows it: running, then done.
#[derive(Clone, Debug, PartialEq)]
pub struct StageView {
    pub kind: StageKind,
    pub label: String,
    /// A decision's branches.
    pub choices: Vec<String>,
    /// What it is doing now while it runs, then what it produced.
    pub detail: String,
    /// The branch a decision took.
    pub chosen: Option<String>,
    /// `None` while it runs.
    pub ok: Option<bool>,
}

/// What the feedback bubble by the tray icon shows about a take.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Feedback {
    pub take: u64,
    pub live: bool,
    /// What the take is doing, such as "Listening".
    pub status: String,
    /// The phrases finished so far, or the transcript once the take is transcribed.
    pub heard: String,
    /// Words of the phrase being spoken.
    pub partial: String,
    /// The decisions, investigations, generations and tool calls, in order.
    pub stages: Vec<StageView>,
    /// Transcription or the stages after it are running: the bubble animates.
    pub working: bool,
    /// The animation's frame, advanced while the take works.
    pub frame: u64,
    /// The text a generation produced.
    pub output: String,
    /// The output is an answer for the user to read, not text typed into the application.
    pub answer: bool,
    /// A tool call waiting for the user's confirmation.
    pub confirm: Option<PendingCall>,
    /// The window the take started in, where Insert types the answer.
    pub window: Option<u64>,
    /// The take ended; the bubble hides shortly.
    pub done: bool,
    pub failed: bool,
    /// A message of its own (recording, automations): it shows even with live feedback off.
    pub message: bool,
    /// Which message this is, so an older message's timer never hides a newer one.
    pub shown: u64,
}

impl Feedback {
    /// Applies a progress update; returns whether it changed what the bubble shows.
    fn apply(&mut self, update: &Update) -> bool {
        match update {
            Update::Level(_) => return false,
            Update::Delta(text) => self.partial.push_str(text),
            Update::Heard(text) => {
                self.heard = text.clone();
                self.partial.clear();
            }
            Update::Transcribing => {
                self.status = "Transcribing…".into();
                self.working = true;
            }
            Update::Thinking => {
                self.status = "Thinking…".into();
                self.working = true;
                if !self.live && self.heard.is_empty() {
                    self.heard = std::mem::take(&mut self.partial);
                }
            }
            Update::Stage(stage) => {
                self.working = true;
                self.status = match stage.kind {
                    StageKind::Deciding => "Deciding…".to_string(),
                    StageKind::Investigating => "Reading the screen…".into(),
                    StageKind::Writing => "Writing…".into(),
                    StageKind::Answering => "Answering…".into(),
                    StageKind::Calling => format!("Calling {}…", stage.label),
                    StageKind::Agent => "Working with tools…".into(),
                };
                self.stages.push(StageView {
                    kind: stage.kind,
                    label: stage.label.clone(),
                    choices: stage.choices.clone(),
                    detail: String::new(),
                    chosen: None,
                    ok: None,
                });
            }
            Update::Progress(now) => {
                if let Some(open) = self.stages.iter_mut().rev().find(|s| s.ok.is_none()) {
                    open.detail = now.clone();
                }
            }
            Update::StageDone { detail, chosen, ok } => {
                if let Some(open) = self.stages.iter_mut().rev().find(|s| s.ok.is_none()) {
                    open.detail = detail.clone();
                    open.chosen = chosen.clone();
                    open.ok = Some(*ok);
                }
            }
            Update::Output(text) => self.output.push_str(text),
            Update::Answering => {
                self.answer = true;
                self.output.clear();
            }
        }
        true
    }

    /// Whether the bubble animates: the take works and has not ended.
    pub fn animating(&self) -> bool {
        self.working && !self.done && self.confirm.is_none()
    }

    /// The take's outcome.
    fn finish(&mut self, trace: &Trace) {
        self.done = true;
        self.working = false;
        for open in self.stages.iter_mut().filter(|s| s.ok.is_none()) {
            open.ok = Some(trace.error.is_none());
        }
        self.partial.clear();
        if !trace.transcript.is_empty() {
            self.heard = trace.transcript.clone();
        }
        self.output = if trace.output != trace.transcript {
            trace.output.clone()
        } else {
            String::new()
        };
        self.failed = trace.error.is_some();
        self.answer = trace.delivery == Some(DeliveryOutcome::Shown);
        if self.answer {
            self.output = trace.output.clone();
        }
        self.status = match (&trace.error, &trace.delivery) {
            (Some(error), _) => error.clone(),
            (None, Some(DeliveryOutcome::Delivered { .. })) => "Inserted".into(),
            (None, Some(DeliveryOutcome::OnClipboard { reason })) => {
                format!("On the clipboard: {reason}")
            }
            (None, Some(DeliveryOutcome::Shown)) => "Answer".into(),
            (None, None) if trace.notes.iter().any(|n| n.starts_with("Too short")) => {
                "Too short to hold speech".into()
            }
            (None, None) => "Done".into(),
        };
    }
}

/// What the inspector shows; the agent writes it, the window reads it.
#[derive(Default)]
pub struct View {
    pub tray: TrayState,
    pub dictating: bool,
    pub live_transcript: String,
    pub live_output: String,
    pub context: Option<ContextSnapshot>,
    pub context_error: Option<String>,
    /// The route the window in front takes through the flow tree before any model decision.
    pub route: Vec<FlowStep>,
    pub traces: VecDeque<Trace>,
    /// The flow tree in use: the last one that loaded without errors.
    pub flows: Arc<FlowTree>,
    /// The problems of the flows folder as it is now; empty when it loads cleanly.
    pub flow_errors: Vec<FlowError>,
    /// Notes from preparing the flows folder, such as an `AGENTS.md` jevons no longer updates.
    pub flow_notes: Vec<String>,
    /// The branch every take starts at, chosen from the tray; `None` is the root.
    pub start: Option<String>,
    /// MCP servers that did not start or list their tools.
    pub tool_problems: Vec<String>,
    pub context_paused: bool,
    pub hotkey_error: Option<String>,
    pub notice: Option<String>,
    pub runtime: Option<Status>,
    pub context_backend: &'static str,
    pub sink_backend: &'static str,
    pub devices: Vec<AudioDevice>,
    pub config: DesktopConfig,
    pub config_file: PathBuf,
    /// Set when the window should show itself.
    pub show_window: bool,
    /// Whether the tray icon runs (otherwise the window is the only way in).
    pub tray_running: bool,
    /// Whether the window is on screen.
    pub window_visible: bool,
    /// Whether the window shows the live context, which the agent then reads twice a second.
    pub watch_context: bool,
    /// The feedback bubble's contents while a take runs and shortly after.
    pub feedback: Option<Feedback>,
    /// The Context tab's workbench: the last extract it tried and what it found.
    pub trial: Option<TrialView>,
    /// Whether a trial is under way.
    pub trying: bool,
    /// The last save from the workbench: where it went, or the tree's problems with it.
    pub saved: Option<Result<String, Vec<String>>>,
    /// What the flow tree's extracts read in the window the Context tab shows.
    pub extracts: Option<ExtractsProbe>,
    /// The Context tab's interface browser: the window's tree as far as it was opened.
    pub interface: Option<InterfaceView>,
    /// The automations library: name, description, and whether this version is approved.
    pub automations: Vec<(String, String, bool)>,
    pub quit: bool,
}

/// An extract the Context tab's workbench tries or saves.
#[derive(Clone, Debug, PartialEq)]
pub struct TrialRequest {
    /// The node file that declares it (such as `ask/slack/generate.toml`); `None` for a new one.
    pub file: Option<String>,
    pub name: String,
    pub spec: jevons_desktop_core::flow::spec::ExtractSpec,
    /// Typed rather than asked for: wait for a pause first.
    pub debounce: bool,
}

/// What the workbench's last trial found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrialView {
    pub name: String,
    /// The expression it tried, to tell an answer from an earlier edit.
    pub xpath: String,
    /// The application and title of the window it read.
    pub window: String,
    pub trial: extract::Trial,
}

/// The pause in typing before the workbench tries an edit.
const TRIAL_PAUSE: Duration = Duration::from_millis(350);

/// The workbench's requests: at most one trial runs, and the latest one waiting runs after it.
#[derive(Default)]
struct Workbench {
    latest: Option<TrialRequest>,
    /// Counts typed requests, so only the last one's pause starts a trial.
    due: u64,
    running: bool,
    /// A request arrived while a trial ran.
    queued: bool,
    live: bool,
    /// The window (application and title) the last trial read, for live mode.
    window: Option<String>,
}

/// The interface of the window the Context tab showed when it was loaded, level by level.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InterfaceView {
    /// Counts loads, so reads for a tree that was loaded again are dropped.
    pub generation: u64,
    /// The window it is the interface of; its id is empty until the top is read.
    pub window: WindowEntry,
    /// The window as the Context tab tells windows apart, to notice when it shows another.
    pub key: String,
    /// The levels read, by their parent's id (the window's id for the top).
    pub levels: HashMap<String, interface::Level>,
    /// The elements shown open.
    pub open: HashSet<String>,
    /// The elements whose children are being read (the empty id for the top).
    pub loading: HashSet<String>,
    /// Why the last read failed: its elements no longer match the window, so it is reloaded.
    pub failed: Option<String>,
    /// An "open all below" read its most elements before reaching its depth.
    pub stopped: bool,
    /// What the last reveal or "show focused" could not do.
    pub note: Option<String>,
    /// A reveal is reading the way down to an element.
    pub revealing: bool,
    /// The row to scroll into view once it is laid out (the window takes it).
    pub reveal: Option<String>,
    pub search: Option<InterfaceSearch>,
    pub selected: Option<UiElement>,
    /// The selected element's selectors, or why there are none; `None` while they are found.
    pub selectors: Option<Result<Vec<Candidate>, String>>,
}

/// The interface browser's search: its text, and what it found (`None` while it runs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InterfaceSearch {
    pub query: String,
    pub found: Option<Result<interface::Found, String>>,
}

/// What the interface browser reveals: an element by its way down, an element found by
/// walking up from it, or the snapshot's focus (found the same way).
enum Reveal {
    Path(Vec<String>),
    Element(String),
    Focused(String),
}

/// How deep "open all below" goes, and the most elements it (and "expand to level") reads.
const OPEN_DEPTH: usize = 6;
const OPEN_BUDGET: usize = 1_500;
/// The most elements a search reads, and how long it may take (checked between its reads).
const SEARCH_BUDGET: usize = 5_000;
const SEARCH_DEADLINE: Duration = Duration::from_secs(2);

/// The flow tree's extracts, read in the window the Context tab shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtractsProbe {
    /// The application and title of the window they were read in.
    pub window: String,
    pub readings: Vec<extract::Reading>,
    /// Whether a reading is under way.
    pub reading: bool,
}

pub type SharedView = Arc<Mutex<View>>;

/// A window as the Context tab's readings tell windows apart: its application and title.
pub fn window_key(snapshot: &ContextSnapshot) -> String {
    format!("{} · {}", snapshot.app.process_name, snapshot.window.title)
}

struct Active {
    id: u64,
    /// Live dictation rather than push-to-talk.
    live: bool,
    /// The hotkey that started it.
    source: Option<u32>,
    /// The hotkey is held while speaking: its release ends the take. Otherwise pressing it
    /// again does.
    held: bool,
    capture: Option<Box<dyn CaptureHandle>>,
    finish: Option<oneshot::Sender<()>>,
    /// The pipeline task, aborted when the take is cancelled.
    task: Option<tokio::task::JoinHandle<()>>,
    /// A note for the recording, transcribed only.
    note: bool,
}

/// A demonstration being recorded.
struct ActiveRecording {
    /// Taken when the recording stops.
    session: Arc<Mutex<Option<Session>>>,
    handle: Option<Box<dyn RecordingHandle>>,
    /// Turns what the platform reports into steps (it reads the interface, so it blocks).
    worker: Option<std::thread::JoinHandle<()>>,
}

/// The id feedback bubbles of messages (not takes) use.
const MESSAGE: u64 = u64::MAX;

pub struct Agent {
    config: DesktopConfig,
    config_file: PathBuf,
    runtime: Runtime,
    tray: Option<Tray>,
    context: Box<dyn ContextProvider>,
    inspector: Arc<dyn ContextInspector>,
    /// Where investigations found their answers, reused across takes.
    paths: Arc<Mutex<PathCache>>,
    sink: Arc<Mutex<Box<dyn TextSink>>>,
    audio: Box<dyn AudioSource>,
    /// The tools the settings register, the library's automations among them.
    tools: Arc<ToolHost>,
    /// The automations library, and what runs them.
    automations: Arc<AutomationHost>,
    actor: Arc<dyn UiActor>,
    recorder: Arc<dyn Recorder>,
    recording: Option<ActiveRecording>,
    /// The record hotkey held: which, since when, and whether this press started recording.
    record_press: Option<(u32, std::time::Instant, bool)>,
    /// The automation running outside a take, and its cancel flag.
    running: Option<(String, Arc<AtomicBool>)>,
    /// The automation the recording in progress will replace (Record it again).
    replacing: Option<String>,
    /// How many messages the bubble has shown.
    messages: u64,
    /// The window (and tree) the Context tab's extracts were last read in.
    extracts_read: Option<String>,
    workbench: Workbench,
    /// Selectors are being found for the interface browser; the latest selection waits.
    finding: bool,
    selection: Option<UiElement>,
    /// A search runs in the interface browser; the latest text asked for meanwhile waits.
    searching: bool,
    search_next: Option<String>,
    flows: Arc<FlowTree>,
    hotkeys: std::collections::HashMap<u32, HotkeyAction>,
    /// The reply to the tool call the bubble asks about.
    confirming: Option<oneshot::Sender<bool>>,
    /// Each registered hotkey's accelerator, to swallow its repeats while it is held.
    accelerators: std::collections::HashMap<u32, String>,
    active: Option<Active>,
    next_take: u64,
    view: SharedView,
    repaint: Arc<dyn Fn() + Send + Sync>,
    commands: mpsc::UnboundedSender<Command>,
    _watcher: Option<notify::RecommendedWatcher>,
    _library_watcher: Option<notify::RecommendedWatcher>,
}

/// The platform layers the agent drives.
pub struct Layers {
    pub context: Box<dyn ContextProvider>,
    pub inspector: Arc<dyn ContextInspector>,
    pub actor: Arc<dyn UiActor>,
    pub recorder: Arc<dyn Recorder>,
    pub sink: Box<dyn TextSink>,
    pub audio: Box<dyn AudioSource>,
    pub tray: Option<Tray>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: DesktopConfig,
        config_file: PathBuf,
        layers: Layers,
        runtime: Runtime,
        view: SharedView,
        repaint: Arc<dyn Fn() + Send + Sync>,
        commands: mpsc::UnboundedSender<Command>,
    ) -> Self {
        config.privacy.apply_api_log();
        let flows_dir = config.flows_dir(&config_file);
        let automations = automation_host(&config, &config_file, &layers, &commands);
        let tools = Arc::new(
            ToolHost::new(&config.tools, &config.mcp).with_automations(automations.clone()),
        );
        let (tree, notes) = defaults::open(&flows_dir, &tools.catalog());
        let errors = tree.errors.clone();
        let flows = if tree.is_valid() {
            tree
        } else {
            // Nothing good to keep yet: the built-in tree runs until the folder is fixed.
            FlowTree::load(&defaults::builtin(), &Catalog::default())
        };
        let watcher = watch(&flows_dir, commands.clone(), || Command::ReloadFlows);
        let library_watcher = watch(automations.dir(), commands.clone(), || {
            Command::ReloadAutomations
        });
        let mut agent = Self {
            tools,
            automations,
            actor: layers.actor,
            recorder: layers.recorder,
            recording: None,
            record_press: None,
            running: None,
            replacing: None,
            messages: 0,
            extracts_read: None,
            workbench: Workbench::default(),
            finding: false,
            selection: None,
            searching: false,
            search_next: None,
            flows: Arc::new(flows),
            config,
            config_file,
            runtime,
            tray: layers.tray,
            context: layers.context,
            inspector: layers.inspector,
            paths: Arc::new(Mutex::new(PathCache::open(PathCache::default_file()))),
            sink: Arc::new(Mutex::new(layers.sink)),
            audio: layers.audio,
            hotkeys: std::collections::HashMap::new(),
            confirming: None,
            accelerators: std::collections::HashMap::new(),
            active: None,
            next_take: 1,
            view,
            repaint,
            commands,
            _watcher: watcher,
            _library_watcher: library_watcher,
        };
        {
            let mut view = agent.view.lock().expect("the view lock");
            view.flows = agent.flows.clone();
            view.flow_errors = errors;
            view.flow_notes = notes;
            view.context_backend = agent.context.name();
            view.sink_backend = agent.sink.lock().expect("the sink lock").name();
            view.devices = agent.audio.devices();
            view.config = agent.config.clone();
            view.config_file = agent.config_file.clone();
        }
        if let Some(tray) = &agent.tray {
            tray.hotkeys(agent.bindings());
        }
        agent.runtime.apply(&agent.config, &agent.config_file);
        agent.publish_menu();
        agent.list_tools();
        agent
    }

    /// Starts the MCP servers in the background; their tools then check the flows.
    fn list_tools(&self) {
        let tools = self.tools.clone();
        let commands = self.commands.clone();
        let dir = self.config.flows_dir(&self.config_file);
        tokio::spawn(async move {
            let problems = tools.start().await;
            for problem in &problems {
                tracing::warn!(problem = %problem, "An MCP server is unavailable");
            }
            if let Err(e) = defaults::write_tools_md(&dir, &tools.tools_md()) {
                tracing::warn!(error = %e, "Cannot write TOOLS.md");
            }
            let _ = commands.send(Command::ToolsListed(problems));
        });
    }

    /// Handles commands until Quit.
    pub async fn run(mut self, mut commands: mpsc::UnboundedReceiver<Command>) {
        let mut watch = tokio::time::interval(Duration::from_millis(500));
        watch.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                command = commands.recv() => {
                    let Some(command) = command else { break };
                    if !self.handle(command) {
                        break;
                    }
                }
                _ = watch.tick() => {
                    let watching = {
                        let view = self.view();
                        view.window_visible && view.watch_context
                    };
                    if watching {
                        self.refresh_context(false);
                    }
                }
            }
        }
    }

    fn view(&self) -> std::sync::MutexGuard<'_, View> {
        self.view.lock().expect("the view lock")
    }

    fn repaint(&self) {
        (self.repaint)();
    }

    fn set_tray(&mut self, state: TrayState) {
        self.view().tray = state;
        if let Some(tray) = &mut self.tray {
            tray.set_state(state);
        }
    }

    fn publish_menu(&mut self) {
        let (start, paused) = {
            let view = self.view();
            (view.start.clone(), view.context_paused)
        };
        let live = self.active.as_ref().is_some_and(|a| a.live);
        let menu = MenuModel {
            busy: self.active.is_some() || self.running.is_some(),
            dictating: self.active.as_ref().is_some_and(|a| !a.live),
            live,
            entries: self.flows.entries(),
            start,
            context_paused: paused,
            feedback: self.config.dictation.live_feedback,
            recording: self.recording.is_some(),
            automations: self
                .automations
                .list()
                .into_iter()
                .map(|a| (a.name, a.description, a.approved))
                .collect(),
        };
        self.view().automations = menu.automations.clone();
        if let Some(tray) = &mut self.tray {
            tray.set_menu(&menu);
        }
    }

    /// Returns false to stop.
    fn handle(&mut self, command: Command) -> bool {
        match command {
            Command::Hotkey(HotkeyEvent::Pressed(id)) => match self.hotkeys.get(&id) {
                Some(HotkeyAction::ShowInspector) => {
                    self.view().show_window = true;
                    self.repaint();
                }
                Some(HotkeyAction::Dictate { entry }) => {
                    let entry = entry.clone();
                    let mode = self.config.dictation.hotkey_mode;
                    self.hotkey_pressed(id, entry, false, mode);
                }
                Some(HotkeyAction::LiveDictation) => {
                    let mode = self.config.dictation.live_hotkey_mode;
                    self.hotkey_pressed(id, None, true, mode);
                }
                Some(HotkeyAction::Confirm(yes)) => {
                    let yes = *yes;
                    self.confirmed(yes);
                }
                Some(HotkeyAction::Record) => self.record_pressed(id),
                Some(HotkeyAction::Automation { name }) => {
                    let name = name.clone();
                    self.automation_hotkey(id, name);
                }
                None => {}
            },
            Command::Hotkey(HotkeyEvent::Released(id))
                if self.record_press.is_some_and(|(held, ..)| held == id) =>
            {
                self.record_released();
            }
            Command::Hotkey(HotkeyEvent::Released(id)) => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| a.source == Some(id) && a.held && a.capture.is_some())
                {
                    self.stop_take();
                }
            }
            Command::HotkeysRegistered { actions, errors } => {
                self.accelerators = actions
                    .iter()
                    .map(|(id, _, accelerator)| (*id, accelerator.clone()))
                    .collect();
                self.hotkeys = actions
                    .into_iter()
                    .map(|(id, action, _)| (id, action))
                    .collect();
                self.view().hotkey_error = (!errors.is_empty()).then(|| errors.join("; "));
                self.repaint();
            }
            Command::Menu(menu) => return self.menu(menu),
            Command::Apply(config) => self.apply(*config),
            Command::RefreshContext => self.refresh_context(false),
            Command::CaptureContextIn(delay) => {
                let commands = self.commands.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let _ = commands.send(Command::RefreshContext);
                });
            }
            Command::RecordTree => self.record_tree(),
            Command::TryExtract(request) => {
                let debounce = request.debounce;
                self.workbench.latest = Some(*request);
                self.view().saved = None;
                if debounce {
                    self.workbench.due += 1;
                    let (due, commands) = (self.workbench.due, self.commands.clone());
                    tokio::spawn(async move {
                        tokio::time::sleep(TRIAL_PAUSE).await;
                        let _ = commands.send(Command::TrialDue(due));
                    });
                } else {
                    self.run_trial();
                }
            }
            Command::LiveTrials(on) => self.workbench.live = on,
            Command::TrialDue(due) => {
                if due == self.workbench.due {
                    self.run_trial();
                }
            }
            Command::TrialDone(done) => {
                self.workbench.running = false;
                self.workbench.window = Some(done.window.clone());
                {
                    let mut view = self.view();
                    view.trial = Some(*done);
                    view.trying = false;
                }
                if std::mem::take(&mut self.workbench.queued) {
                    self.run_trial();
                }
                self.repaint();
            }
            Command::SaveExtract(request) => self.save_extract(*request),
            Command::ReadExtracts => self.read_extracts(true),
            Command::InterfaceLoad => self.load_interface(),
            Command::InterfaceOpen { id, all } => self.open_interface(id, all),
            Command::InterfaceClose(id) => {
                if let Some(browser) = self.view().interface.as_mut() {
                    browser.open.remove(&id);
                }
                self.repaint();
            }
            Command::InterfaceMore(id) => self.more_interface(id),
            Command::InterfaceExpand(depth) => self.expand_interface(depth),
            Command::InterfaceCollapse => {
                if let Some(browser) = self.view().interface.as_mut() {
                    browser.open.retain(|id| *id == browser.window.id);
                }
                self.repaint();
            }
            Command::InterfaceSearch(query) => {
                let query = query.trim().to_string();
                if let Some(browser) = self.view().interface.as_mut() {
                    browser.search = (!query.is_empty()).then(|| InterfaceSearch {
                        query: query.clone(),
                        found: None,
                    });
                }
                if query.is_empty() {
                    self.search_next = None;
                } else if self.searching {
                    self.search_next = Some(query);
                } else {
                    self.search_interface(query);
                }
                self.repaint();
            }
            Command::InterfaceSearched(generation, query, found) => {
                self.searching = false;
                if let Some(search) = self
                    .view()
                    .interface
                    .as_mut()
                    .filter(|b| b.generation == generation)
                    .and_then(|b| b.search.as_mut())
                    .filter(|s| s.query == query)
                {
                    search.found = Some(found);
                }
                if let Some(next) = self.search_next.take() {
                    self.search_interface(next);
                }
                self.repaint();
            }
            Command::InterfaceReveal(path) => self.reveal_interface(match path.as_slice() {
                [only] => Reveal::Element(only.clone()),
                _ => Reveal::Path(path),
            }),
            Command::InterfaceFocus => {
                let focused = self
                    .view()
                    .context
                    .as_ref()
                    .and_then(|c| c.focused.as_ref())
                    .and_then(|f| f.id.clone());
                match focused {
                    Some(id) => self.reveal_interface(Reveal::Focused(id)),
                    None => {
                        if let Some(browser) = self.view().interface.as_mut() {
                            browser.note = Some(
                                "The snapshot has no focused element to show (capture again with \
                                 the element focused)"
                                    .into(),
                            );
                        }
                        self.repaint();
                    }
                }
            }
            Command::InterfaceSelect(element) => {
                // Its properties show at once; its selectors once a search is free.
                if let Some(browser) = self.view().interface.as_mut() {
                    browser.selected = Some((*element).clone());
                    browser.selectors = None;
                }
                self.selection = Some(*element);
                self.find_selectors();
                self.repaint();
            }
            Command::InterfaceFound => {
                self.finding = false;
                self.find_selectors();
            }
            Command::ToolsListed(problems) => {
                self.view().tool_problems = problems;
                self.reload_flows();
            }
            Command::ReloadFlows => self.reload_flows(),
            Command::ReloadAutomations => {
                self.automations.reload();
                // The tools changed: rewrite TOOLS.md and check the flows against them.
                self.list_tools();
                self.publish_menu();
            }
            Command::RecordingNote(said) => self.recording_note(said),
            Command::RecordingSaved(saved) => self.recording_saved(saved),
            Command::Authored(authored) => self.authored(authored),
            Command::Reviewed(report) => self.reviewed(*report),
            Command::ApprovalAnswered { name, version, yes } => {
                self.approval_answered(&name, &version, yes)
            }
            Command::AutomationProgress(label) => {
                if let Some(feedback) = self.view().feedback.as_mut().filter(|f| f.take == MESSAGE)
                {
                    feedback.status = label;
                }
                self.repaint();
            }
            Command::AutomationFinished(trace) => self.automation_finished(*trace),
            Command::ReloadRuntime => self.runtime.apply(&self.config, &self.config_file),
            Command::RuntimeChanged => {
                let status = self.runtime.status();
                if self.active.is_none() {
                    let state = match status {
                        Status::Ready { .. } | Status::Remote { .. } => TrayState::Idle,
                        Status::Failed(_) => TrayState::Error,
                        Status::Loading => TrayState::Loading,
                        _ => TrayState::Offline,
                    };
                    self.set_tray(state);
                }
                self.view().runtime = Some(status);
                self.repaint();
            }
            Command::TakeFinished(trace) => self.finished(*trace),
            Command::ConfirmRequested(confirmation) => self.confirm_requested(confirmation),
            Command::Confirmed(yes) => self.confirmed(yes),
            Command::Bubble(action) => self.bubble(action),
            Command::HideMessage(shown) => {
                let mut view = self.view();
                if view
                    .feedback
                    .as_ref()
                    .is_some_and(|f| f.take == MESSAGE && f.shown == shown && f.done)
                {
                    view.feedback = None;
                    drop(view);
                    self.repaint();
                }
            }
            Command::HideFeedback(take) => {
                let mut view = self.view();
                if view
                    .feedback
                    .as_ref()
                    .is_some_and(|f| f.take == take && f.done)
                {
                    view.feedback = None;
                    drop(view);
                    self.repaint();
                }
            }
        }
        true
    }

    fn menu(&mut self, command: MenuCommand) -> bool {
        match command {
            MenuCommand::ToggleDictation => self.toggle(),
            MenuCommand::ToggleLiveDictation => self.toggle_live(),
            MenuCommand::CancelTake => {
                if let Some(active) = &self.active {
                    tracing::info!(take = active.id, "Take cancelled");
                }
                if let Some((name, cancel)) = &self.running {
                    tracing::info!(automation = %name, "Automation cancelled");
                    cancel.store(true, Ordering::Relaxed);
                }
                self.cancel_take();
                self.view().dictating = false;
                self.notice("The take was cancelled");
                self.set_tray(TrayState::Idle);
                self.publish_menu();
            }
            MenuCommand::OpenLogsFolder => {
                let dir = jevons_desktop_core::config::user_dir();
                let _ = std::fs::create_dir_all(dir.join("traces"));
                open_folder(&dir);
            }
            MenuCommand::StartAt(branch) => {
                self.view().start = branch;
                self.publish_menu();
                self.refresh_context(true);
            }
            MenuCommand::ShowInspector => {
                self.view().show_window = true;
                self.repaint();
            }
            MenuCommand::ToggleFeedback => {
                let mut config = self.config.clone();
                config.dictation.live_feedback = !config.dictation.live_feedback;
                if let Err(e) = config.save(&self.config_file) {
                    self.notice(&e.to_string());
                } else {
                    self.config = config;
                    self.view().config = self.config.clone();
                    self.publish_menu();
                    self.repaint();
                }
            }
            MenuCommand::ToggleContextPause => {
                {
                    let mut view = self.view();
                    view.context_paused = !view.context_paused;
                }
                self.publish_menu();
                self.repaint();
            }
            MenuCommand::ReloadFlows => self.reload_flows(),
            MenuCommand::ToggleRecording => {
                if self.recording.is_some() {
                    self.stop_recording();
                } else {
                    self.start_recording();
                }
            }
            MenuCommand::OpenAutomationsFolder => {
                let dir = self.automations.dir().to_path_buf();
                let _ = std::fs::create_dir_all(&dir);
                open_folder(&dir);
            }
            MenuCommand::RunAutomation(name) => self.run_automation(&name, None, false),
            MenuCommand::RunStepByStep(name) => self.run_automation(&name, None, true),
            MenuCommand::RecordAgain(name) => {
                self.replacing = Some(name);
                self.start_recording();
                if self.recording.is_none() {
                    self.replacing = None;
                }
            }
            MenuCommand::ApproveAutomation(name) => self.review_automation(&name),
            MenuCommand::OpenConfigFolder => {
                if let Some(dir) = self.config_file.parent() {
                    let _ = std::fs::create_dir_all(dir);
                    open_folder(dir);
                }
            }
            MenuCommand::Quit => {
                self.cancel_take();
                if let Some(tray) = &self.tray {
                    tray.quit();
                }
                self.runtime.shutdown();
                self.view().quit = true;
                self.repaint();
                return false;
            }
        }
        true
    }

    /// A dictation hotkey went down: it starts a take, or stops the one it started in toggle
    /// mode. Key repeats and other hotkeys while a take runs change nothing.
    fn hotkey_pressed(&mut self, id: u32, entry: Option<String>, live: bool, mode: HotkeyMode) {
        match &self.active {
            None => self.begin(entry, Some(id), live, mode == HotkeyMode::Hold, None),
            Some(a) if a.source == Some(id) && !a.held && a.capture.is_some() => self.stop_take(),
            Some(_) => {}
        }
    }

    /// Starts or stops push-to-talk dictation from the menu.
    fn toggle(&mut self) {
        match &self.active {
            Some(active) if !active.live && active.capture.is_some() => self.stop_take(),
            Some(active) if active.live => self.notice("Live dictation is running"),
            Some(_) => self.notice("The last take is still being processed"),
            None => self.begin(None, None, false, false, None),
        }
    }

    fn toggle_live(&mut self) {
        match &self.active {
            Some(active) if active.live && active.capture.is_some() => self.stop_take(),
            Some(active) if active.live => {}
            Some(_) => self.notice("Finish the current take first"),
            None => self.begin(None, None, true, false, None),
        }
    }

    fn notice(&mut self, message: &str) {
        self.view().notice = Some(message.into());
        self.repaint();
    }

    fn snapshot(&self) -> Result<ContextSnapshot, String> {
        if self.view().context_paused {
            return Ok(ContextSnapshot {
                errors: vec!["Context capture is paused".into()],
                ..ContextSnapshot::default()
            });
        }
        self.context
            .snapshot(&self.config.privacy)
            .map_err(|e| e.to_string())
    }

    fn refresh_context(&mut self, keep_snapshot: bool) {
        let snapshot = if keep_snapshot {
            self.view().context.clone().ok_or_else(String::new)
        } else {
            self.snapshot()
        };
        // Reading the context while the inspector has focus would describe the inspector.
        if let Ok(s) = &snapshot
            && s.app.pid == Some(std::process::id())
        {
            return;
        }
        let start = self.view().start.clone();
        let route = snapshot
            .as_ref()
            .map(|s| self.preview(s, start.as_deref()))
            .unwrap_or_default();
        let mut view = self.view();
        // A capture freezes the tab, so the extracts follow any new snapshot while the window is
        // open (reading again only when the window or the tree changed).
        let open = view.window_visible;
        match snapshot {
            Ok(snapshot) => {
                view.route = route;
                view.context = Some(snapshot);
                view.context_error = None;
            }
            Err(e) if !e.is_empty() => view.context_error = Some(e),
            Err(_) => {}
        }
        drop(view);
        if open {
            self.read_extracts(false);
            // Live mode follows the window: a new one (or a new title) is tried again.
            let window = self.view().context.as_ref().map(window_key);
            if self.workbench.live
                && self.workbench.latest.is_some()
                && window.is_some()
                && window != self.workbench.window
            {
                self.run_trial();
            }
        }
        self.repaint();
    }

    /// Tries the workbench's latest extract on the window the Context tab shows, off the agent
    /// thread; one at a time, the latest request waiting for the one running.
    fn run_trial(&mut self) {
        let Some(request) = self.workbench.latest.clone() else {
            return;
        };
        if self.workbench.running {
            self.workbench.queued = true;
            return;
        }
        let Some(snapshot) = self.view().context.clone() else {
            return;
        };
        self.workbench.running = true;
        // Live mode would otherwise try again at once for a window it is already reading.
        self.workbench.window = Some(window_key(&snapshot));
        self.view().trying = true;
        let tree = self.flows.clone();
        let reader = Reader::new(self.inspector.clone(), self.config.privacy.clone());
        let commands = self.commands.clone();
        tokio::task::spawn_blocking(move || {
            let name = if request.name.trim().is_empty() {
                "new".to_string()
            } else {
                request.name.trim().to_string()
            };
            let trial = extract::trial(&tree, &name, &request.spec, &snapshot, &reader);
            let _ = commands.send(Command::TrialDone(Box::new(TrialView {
                name,
                xpath: request.spec.xpath.clone(),
                window: window_key(&snapshot),
                trial,
            })));
        });
        self.repaint();
    }

    /// Reads the top of the interface of the window the Context tab shows, replacing the tree,
    /// off the agent thread.
    fn load_interface(&mut self) {
        let Some(snapshot) = self.view().context.clone() else {
            return;
        };
        let generation = {
            let mut view = self.view();
            let generation = view.interface.as_ref().map_or(0, |b| b.generation) + 1;
            view.interface = Some(InterfaceView {
                generation,
                key: window_key(&snapshot),
                loading: HashSet::from([String::new()]),
                ..InterfaceView::default()
            });
            generation
        };
        self.selection = None;
        let inspector = self.inspector.clone();
        let privacy = self.config.privacy.clone();
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let read = interface::window(&*inspector, &snapshot, &privacy).and_then(|window| {
                let level = interface::level(&*inspector, &window.id).map_err(|e| e.to_string());
                level.map(|level| (window, level))
            });
            let mut view = view.lock().expect("the view lock");
            if let Some(browser) = view
                .interface
                .as_mut()
                .filter(|b| b.generation == generation)
            {
                browser.loading.clear();
                match read {
                    Ok((window, level)) => {
                        browser.levels.insert(window.id.clone(), level);
                        browser.open.insert(window.id.clone());
                        browser.window = window;
                    }
                    Err(e) => browser.failed = Some(e),
                }
            }
            drop(view);
            repaint();
        });
        self.repaint();
    }

    /// Opens an element of the interface browser, reading its children (or, with `all`,
    /// everything below it to a depth) off the agent thread, one read per element at a time.
    fn open_interface(&mut self, id: String, all: bool) {
        let (generation, known) = {
            let mut view = self.view();
            let Some(browser) = view.interface.as_mut() else {
                return;
            };
            browser.open.insert(id.clone());
            if browser.loading.contains(&id) || (!all && browser.levels.contains_key(&id)) {
                drop(view);
                self.repaint();
                return;
            }
            browser.loading.insert(id.clone());
            let known = if all {
                browser.levels.clone()
            } else {
                HashMap::new()
            };
            (browser.generation, known)
        };
        let inspector = self.inspector.clone();
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let read = if all {
                interface::open_below(&*inspector, &id, OPEN_DEPTH, OPEN_BUDGET, &known)
            } else {
                interface::level(&*inspector, &id).map(|level| interface::Opened {
                    levels: vec![(id.clone(), level)],
                    stopped: false,
                })
            };
            let mut view = view.lock().expect("the view lock");
            if let Some(browser) = view
                .interface
                .as_mut()
                .filter(|b| b.generation == generation)
            {
                browser.loading.remove(&id);
                match read {
                    Ok(opened) => {
                        browser.stopped = opened.stopped;
                        for (parent, level) in opened.levels {
                            if all {
                                browser.open.insert(parent.clone());
                            }
                            browser.levels.insert(parent, level);
                        }
                    }
                    Err(e) => {
                        browser.open.remove(&id);
                        browser.failed = Some(format!(
                            "Its elements no longer match the window ({e}): reload the tree"
                        ));
                    }
                }
            }
            drop(view);
            repaint();
        });
        self.repaint();
    }

    /// Reads the next page of an element's children, off the agent thread, appending it when
    /// the level is still as it was.
    fn more_interface(&mut self, id: String) {
        let (generation, offset) = {
            let mut view = self.view();
            let Some(browser) = view.interface.as_mut() else {
                return;
            };
            let Some(offset) = browser.levels.get(&id).map(|l| l.elements.len()) else {
                return;
            };
            if !browser.loading.insert(id.clone()) {
                return;
            }
            (browser.generation, offset)
        };
        let inspector = self.inspector.clone();
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let read = interface::page(&*inspector, &id, offset, interface::PER_LEVEL);
            let mut view = view.lock().expect("the view lock");
            if let Some(browser) = view
                .interface
                .as_mut()
                .filter(|b| b.generation == generation)
            {
                browser.loading.remove(&id);
                match read {
                    Ok(page) => {
                        if let Some(level) = browser
                            .levels
                            .get_mut(&id)
                            .filter(|l| l.elements.len() == offset)
                        {
                            level.elements.extend(page.elements);
                            level.total = page.total;
                        }
                    }
                    Err(e) => {
                        browser.failed = Some(format!(
                            "Its elements no longer match the window ({e}): reload the tree"
                        ));
                    }
                }
            }
            drop(view);
            repaint();
        });
        self.repaint();
    }

    /// Opens the interface from the window down to `depth` levels, reading the levels it lacks
    /// (at most [`OPEN_BUDGET`] elements) off the agent thread; everything deeper closes.
    fn expand_interface(&mut self, depth: usize) {
        let (generation, window, known) = {
            let mut view = self.view();
            let Some(browser) = view.interface.as_mut() else {
                return;
            };
            let window = browser.window.id.clone();
            if window.is_empty() || !browser.loading.insert(window.clone()) {
                return;
            }
            (browser.generation, window, browser.levels.clone())
        };
        let inspector = self.inspector.clone();
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let read = interface::open_below(&*inspector, &window, depth, OPEN_BUDGET, &known);
            let mut view = view.lock().expect("the view lock");
            if let Some(browser) = view
                .interface
                .as_mut()
                .filter(|b| b.generation == generation)
            {
                browser.loading.remove(&window);
                match read {
                    Ok(opened) => {
                        browser.stopped = opened.stopped;
                        browser.open.clear();
                        for (parent, level) in opened.levels {
                            browser.open.insert(parent.clone());
                            browser.levels.insert(parent, level);
                        }
                    }
                    Err(e) => {
                        browser.failed = Some(format!(
                            "The window no longer matches the tree ({e}): reload it"
                        ));
                    }
                }
            }
            drop(view);
            repaint();
        });
        self.repaint();
    }

    /// Searches the whole window of the interface browser for `query`, off the agent thread.
    fn search_interface(&mut self, query: String) {
        let Some((generation, window)) = self
            .view()
            .interface
            .as_ref()
            .filter(|b| !b.window.id.is_empty())
            .map(|b| (b.generation, b.window.id.clone()))
        else {
            return;
        };
        self.searching = true;
        let inspector = self.inspector.clone();
        let commands = self.commands.clone();
        tokio::task::spawn_blocking(move || {
            let found =
                interface::search(&*inspector, &window, &query, SEARCH_BUDGET, SEARCH_DEADLINE)
                    .map_err(|e| {
                        format!("The window could not be searched ({e}): reload the tree")
                    });
            let _ = commands.send(Command::InterfaceSearched(generation, query, found));
        });
    }

    /// Opens the interface browser down to an element and selects it, reading the levels on
    /// the way off the agent thread; the window then scrolls its row into view.
    fn reveal_interface(&mut self, target: Reveal) {
        let (generation, window, known) = {
            let mut view = self.view();
            let Some(browser) = view.interface.as_mut().filter(|b| !b.window.id.is_empty()) else {
                return;
            };
            if browser.revealing {
                return;
            }
            browser.revealing = true;
            browser.note = None;
            (
                browser.generation,
                browser.window.id.clone(),
                browser.levels.clone(),
            )
        };
        let inspector = self.inspector.clone();
        let view = self.view.clone();
        let commands = self.commands.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let (id, what, again) = match target {
                Reveal::Path(path) => (Err(path), "", ""),
                Reveal::Element(id) => (Ok(id), "The element", "search again or reload the tree"),
                Reveal::Focused(id) => (
                    Ok(id),
                    "The focused element",
                    "reload the tree or capture again",
                ),
            };
            let path = match id {
                Err(path) => Ok(path),
                Ok(id) => interface::ancestry(&*inspector, &id)
                    .map_err(|e| format!("{what} is gone ({e}): {again}"))
                    .and_then(|chain| match chain.split_first() {
                        Some((top, path)) if *top == window => Ok(path.to_vec()),
                        _ => Err(format!(
                            "{what} is in another window than this tree's: reload the tree"
                        )),
                    }),
            };
            let revealed = path.and_then(|path| {
                interface::reveal(&*inspector, &known, &window, &path)
                    .map(|r| (path.len(), r))
                    .map_err(|e| format!("The window no longer matches the tree ({e}): reload it"))
            });
            let mut chosen = None;
            let mut guard = view.lock().expect("the view lock");
            if let Some(browser) = guard
                .interface
                .as_mut()
                .filter(|b| b.generation == generation)
            {
                browser.revealing = false;
                match revealed {
                    Ok((asked, revealed)) => {
                        for (parent, level) in revealed.levels {
                            browser.levels.insert(parent, level);
                        }
                        browser.open.insert(window.clone());
                        let reached = revealed.reached.len();
                        for (i, element) in revealed.reached.iter().enumerate() {
                            if i + 1 < reached {
                                browser.open.insert(element.id.clone());
                            }
                        }
                        if reached < asked {
                            browser.note = Some(
                                "The element is no longer in the window (its nearest ancestor is \
                                 selected): reload the tree"
                                    .into(),
                            );
                        }
                        chosen = revealed.reached.last().cloned();
                        browser.reveal = chosen.as_ref().map(|e| e.id.clone());
                    }
                    Err(note) => browser.note = Some(note),
                }
            }
            drop(guard);
            if let Some(element) = chosen {
                let _ = commands.send(Command::InterfaceSelect(Box::new(element)));
            }
            repaint();
        });
        self.repaint();
    }

    /// Finds the selectors of the element selected in the interface browser, off the agent
    /// thread: one search at a time, the latest selection waiting for the one running.
    fn find_selectors(&mut self) {
        if self.finding {
            return;
        }
        let Some(element) = self.selection.take() else {
            return;
        };
        let (generation, window) = match self.view().interface.as_ref() {
            Some(browser) => (browser.generation, browser.window.clone()),
            None => return,
        };
        self.finding = true;
        let inspector = self.inspector.clone();
        let view = self.view.clone();
        let commands = self.commands.clone();
        tokio::task::spawn_blocking(move || {
            let found = interface::selectors(&*inspector, &window, &element.id);
            if let Some(browser) = view
                .lock()
                .expect("the view lock")
                .interface
                .as_mut()
                .filter(|b| {
                    b.generation == generation
                        && b.selected.as_ref().is_some_and(|s| s.id == element.id)
                })
            {
                browser.selectors = Some(found);
            }
            let _ = commands.send(Command::InterfaceFound);
        });
        self.repaint();
    }

    /// Writes an edited extract into its node file in the flows folder, off the agent thread,
    /// once the tree with the change still loads; the folder watcher then reloads it.
    fn save_extract(&mut self, request: TrialRequest) {
        let Some(file) = request.file.clone() else {
            return;
        };
        let dir = self.config.flows_dir(&self.config_file);
        let catalog = self.tools.catalog();
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let saved = extract::save(&dir, &file, &request.name, &request.spec, &catalog)
                .map(|()| format!("Saved [extract.{}] into {file}", request.name));
            view.lock().expect("the view lock").saved = Some(saved);
            repaint();
        });
    }

    /// Reads the flow tree's extracts in the window the Context tab shows, off the agent
    /// thread: when that window (or the tree) changed since the last reading, or on `force`.
    fn read_extracts(&mut self, force: bool) {
        let Some(snapshot) = self.view().context.clone() else {
            return;
        };
        let key = format!(
            "{}\u{1f}{}\u{1f}{:p}",
            snapshot.app.process_name,
            snapshot.window.title,
            Arc::as_ptr(&self.flows)
        );
        if !force && self.extracts_read.as_deref() == Some(key.as_str()) {
            return;
        }
        {
            let mut view = self.view();
            let probe = view.extracts.get_or_insert_with(ExtractsProbe::default);
            if probe.reading {
                return;
            }
            probe.reading = true;
        }
        self.extracts_read = Some(key);
        let tree = self.flows.clone();
        let reader = Reader::new(self.inspector.clone(), self.config.privacy.clone());
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let readings = extract::read_applicable(&tree, &snapshot, &reader);
            view.lock().expect("the view lock").extracts = Some(ExtractsProbe {
                window: window_key(&snapshot),
                readings,
                reading: false,
            });
            repaint();
        });
        self.repaint();
    }

    /// The route `snapshot` takes from `start` (or the root) before any model decision.
    fn preview(&self, snapshot: &ContextSnapshot, start: Option<&str>) -> Vec<FlowStep> {
        let entry = start
            .and_then(|s| self.flows.find(s))
            .unwrap_or_else(|| self.flows.root());
        walk::preview(&self.flows, snapshot, entry)
    }

    /// Every hotkey the settings define, plus Enter and Esc while a tool call waits.
    fn bindings(&self) -> Vec<Binding> {
        let mut bindings = Binding::from_settings(&self.config.dictation);
        bindings.extend(Binding::for_automations(&self.config.automation));
        if self.confirming.is_some() {
            bindings.push(Binding {
                accelerator: "Enter".into(),
                action: HotkeyAction::Confirm(true),
            });
            bindings.push(Binding {
                accelerator: "Escape".into(),
                action: HotkeyAction::Confirm(false),
            });
        }
        bindings
    }

    /// Shows a tool call in the bubble and waits for the user.
    fn confirm_requested(&mut self, confirmation: Confirmation) {
        if let Some(previous) = self.confirming.take() {
            let _ = previous.send(false);
        }
        let arguments = serde_json::to_string_pretty(&confirmation.arguments).unwrap_or_default();
        tracing::info!(tool = %confirmation.tool, "Waiting for confirmation");
        self.confirming = Some(confirmation.reply);
        {
            let mut view = self.view();
            // Automations run and are approved outside takes: they get a bubble of their own.
            let feedback = view.feedback.get_or_insert_with(|| Feedback {
                take: MESSAGE,
                message: true,
                ..Feedback::default()
            });
            feedback.status = if confirmation.tool.starts_with("Approve ") {
                "Enter approves this version, Esc keeps it as a draft".into()
            } else {
                "Run this? Enter runs it, Esc cancels".into()
            };
            feedback.done = false;
            feedback.confirm = Some(PendingCall {
                tool: confirmation.tool,
                arguments,
            });
        }
        if let Some(tray) = &self.tray {
            tray.hotkeys(self.bindings());
        }
        self.repaint();
    }

    fn confirmed(&mut self, yes: bool) {
        let Some(reply) = self.confirming.take() else {
            return;
        };
        tracing::info!(approved = yes, "Confirmation answered");
        let _ = reply.send(yes);
        if let Some(feedback) = self.view().feedback.as_mut() {
            feedback.confirm = None;
            feedback.status = if yes {
                "Running the tool…"
            } else {
                "Cancelled"
            }
            .into();
        }
        if let Some(tray) = &self.tray {
            tray.hotkeys(self.bindings());
        }
        self.repaint();
    }

    /// The bubble's answer buttons.
    fn bubble(&mut self, action: BubbleAction) {
        let Some(feedback) = self.view().feedback.take() else {
            return;
        };
        self.repaint();
        match action {
            BubbleAction::Close => {}
            BubbleAction::Copy => {
                let copied = self
                    .sink
                    .lock()
                    .expect("the sink lock")
                    .copy(&feedback.output);
                self.notice(&match copied {
                    Ok(()) => "The answer is on the clipboard".to_string(),
                    Err(e) => e.to_string(),
                });
            }
            BubbleAction::Insert => {
                let sink = self.sink.clone();
                let view = self.view.clone();
                let repaint = self.repaint.clone();
                tokio::spawn(async move {
                    // Closing the bubble gives the focus back to the application underneath.
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    let request = jevons_desktop_core::platform::DeliveryRequest {
                        action: jevons_desktop_core::platform::Action::Insert,
                        text: feedback.output.clone(),
                        method: jevons_desktop_core::platform::DeliveryMethod::Paste,
                        select_all: false,
                        erase: 0,
                    };
                    let outcome = pipeline::deliver_text(
                        &sink,
                        feedback.take,
                        feedback.window.unwrap_or(0),
                        request,
                    )
                    .await;
                    view.lock().expect("the view lock").notice = match outcome {
                        Ok(Some(DeliveryOutcome::OnClipboard { reason })) => {
                            Some(format!("The answer is on the clipboard: {reason}"))
                        }
                        Err(e) => Some(e),
                        _ => None,
                    };
                    repaint();
                });
            }
        }
    }

    /// Saves the front window's interface to `~/jevons/trees`, off the agent thread.
    fn record_tree(&mut self) {
        let inspector = self.inspector.clone();
        let app = self
            .view()
            .context
            .as_ref()
            .map(|c| c.app.process_name.clone());
        let view = self.view.clone();
        let repaint = self.repaint.clone();
        tokio::task::spawn_blocking(move || {
            let result = inspector.windows().and_then(|windows| {
                let window = windows
                    .iter()
                    .find(|w| {
                        app.as_deref()
                            .is_some_and(|a| w.app.eq_ignore_ascii_case(a))
                    })
                    .or_else(|| windows.iter().find(|w| w.front))
                    .cloned()
                    .ok_or_else(|| {
                        jevons_desktop_core::platform::PlatformError::Failed(
                            "no window to record".into(),
                        )
                    })?;
                let tree = RecordedTree::record(
                    inspector.as_ref(),
                    std::slice::from_ref(&window),
                    40,
                    5000,
                )?;
                let dir = jevons_desktop_core::config::user_dir().join("trees");
                let millis = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis());
                let file = dir.join(format!("{millis}-{}.json", window.app));
                std::fs::create_dir_all(&dir)
                    .and_then(|()| {
                        std::fs::write(&file, serde_json::to_vec_pretty(&tree).unwrap_or_default())
                    })
                    .map_err(|e| {
                        jevons_desktop_core::platform::PlatformError::Failed(e.to_string())
                    })?;
                Ok(file)
            });
            view.lock().expect("the view lock").notice = Some(match result {
                Ok(file) => format!("Recorded the interface in {}", file.display()),
                Err(e) => format!("Cannot record the interface: {e}"),
            });
            repaint();
        });
    }

    /// Loads the flows folder again; a tree with errors is reported and the last good one kept.
    fn reload_flows(&mut self) {
        let dir = self.config.flows_dir(&self.config_file);
        let (tree, notes) = defaults::open(&dir, &self.tools.catalog());
        let errors = tree.errors.clone();
        if tree.is_valid() {
            tracing::info!(nodes = tree.nodes().len(), "Flow tree loaded");
            self.flows = Arc::new(tree);
        } else {
            tracing::warn!(
                errors = errors.len(),
                "The flow tree has errors: keeping the last good one"
            );
        }
        {
            let mut view = self.view();
            view.flows = self.flows.clone();
            view.flow_errors = errors;
            view.flow_notes = notes;
        }
        self.publish_menu();
        self.refresh_context(true);
    }

    fn apply(&mut self, config: DesktopConfig) {
        if let Err(e) = config.save(&self.config_file) {
            self.notice(&e.to_string());
            return;
        }
        let hotkeys_changed = Binding::from_settings(&config.dictation)
            != Binding::from_settings(&self.config.dictation)
            || Binding::for_automations(&config.automation)
                != Binding::for_automations(&self.config.automation);
        let flows_changed =
            config.flows_dir(&self.config_file) != self.config.flows_dir(&self.config_file);
        let tools_changed = config.tools != self.config.tools || config.mcp != self.config.mcp;
        let library_changed = config.automations_dir(&self.config_file)
            != self.config.automations_dir(&self.config_file);
        self.config = config;
        self.config.privacy.apply_api_log();
        self.automations
            .set_settings(self.config.automation.clone());
        if hotkeys_changed && let Some(tray) = &self.tray {
            tray.hotkeys(self.bindings());
        }
        if flows_changed {
            let dir = self.config.flows_dir(&self.config_file);
            let _ = std::fs::create_dir_all(&dir);
            self._watcher = watch(&dir, self.commands.clone(), || Command::ReloadFlows);
            self.reload_flows();
        }
        if library_changed {
            let layers = (self.inspector.clone(), self.actor.clone());
            self.automations =
                automation_host_over(&self.config, &self.config_file, layers, &self.commands);
            self._library_watcher = watch(self.automations.dir(), self.commands.clone(), || {
                Command::ReloadAutomations
            });
        }
        if tools_changed || flows_changed || library_changed {
            self.tools = Arc::new(
                ToolHost::new(&self.config.tools, &self.config.mcp)
                    .with_automations(self.automations.clone()),
            );
            self.list_tools();
        }
        self.runtime.apply(&self.config, &self.config_file);
        {
            let mut view = self.view();
            view.config = self.config.clone();
            view.notice = Some(format!("Settings saved to {}", self.config_file.display()));
        }
        self.repaint();
    }

    /// Starts a take: push-to-talk, or live dictation when `live`. `source` is the hotkey that
    /// started it; when `held`, its release ends the take.
    fn begin(
        &mut self,
        entry: Option<String>,
        source: Option<u32>,
        live: bool,
        held: bool,
        flows: Option<Arc<FlowTree>>,
    ) {
        let Some(connection) = self.runtime.connection() else {
            let status = self.runtime.status().describe();
            self.notice(&format!("Dictation is unavailable: {status}"));
            self.set_tray(TrayState::Error);
            return;
        };
        let id = self.next_take;
        self.next_take += 1;
        // Read the context first, before the user's focus can move.
        let context = self.snapshot().unwrap_or_default();
        let (audio, audio_events) = mpsc::unbounded_channel();
        let capture = match self
            .audio
            .start(self.config.dictation.microphone.as_deref(), audio)
        {
            Ok(capture) => capture,
            Err(e) => {
                self.notice(&format!("Cannot open the microphone: {e}"));
                self.set_tray(TrayState::Error);
                return;
            }
        };
        let (finish, finished) = oneshot::channel();
        let dictation = &self.config.dictation;
        let (confirm, mut asked) = mpsc::unbounded_channel::<Confirmation>();
        let forward = self.commands.clone();
        tokio::spawn(async move {
            while let Some(confirmation) = asked.recv().await {
                let _ = forward.send(Command::ConfirmRequested(confirmation));
            }
        });
        let investigator = connection.models.generative.clone().map(|model| {
            Arc::new(Investigator::new(
                connection.client.clone(),
                model,
                self.inspector.clone(),
                self.config.privacy.clone(),
                self.paths.clone(),
            )) as Arc<dyn Investigate>
        });
        let env = Env {
            client: connection.client,
            flows: flows.unwrap_or_else(|| self.flows.clone()),
            settings: pipeline::Settings {
                models: connection.models,
                realtime: connection.realtime,
                language: dictation.language.clone(),
                decide: dictation.decide,
                max_output_tokens: dictation.max_output_tokens,
                ..pipeline::Settings::default()
            },
            sink: Some(self.sink.clone()),
            investigator,
            reader: Some(Arc::new(Reader::new(
                self.inspector.clone(),
                self.config.privacy.clone(),
            ))),
            confirmer: Some(Arc::new(ChannelConfirmer::new(confirm))),
            tools: Some(self.tools.clone()),
        };
        let start = TakeStart {
            id,
            context: context.clone(),
            entry: entry.or_else(|| self.view().start.clone()),
        };
        let route = self.preview(&context, start.entry.as_deref());
        {
            let mut view = self.view();
            view.dictating = true;
            view.live_transcript.clear();
            view.live_output.clear();
            view.notice = None;
            view.feedback = Some(Feedback {
                take: id,
                live,
                window: context.window.handle,
                status: format!(
                    "{}: {}",
                    if live { "Live dictation" } else { "Listening" },
                    match (source, held) {
                        (Some(_), true) => "release the hotkey to finish",
                        (Some(_), false) => "press the hotkey again to finish",
                        (None, _) => "stop it from the tray menu",
                    }
                ),
                ..Feedback::default()
            });
            view.route = route;
            view.context = Some(context);
        }
        self.active = Some(Active {
            id,
            live,
            source,
            held,
            capture: Some(capture),
            finish: Some(finish),
            task: None,
            note: false,
        });
        // Keep the held hotkey's repeats out of the focused application while the take runs.
        if held && let Some(accelerator) = source.and_then(|id| self.accelerators.get(&id)) {
            crate::hold::hold(accelerator);
        }
        self.set_tray(TrayState::Listening { level: 0 });
        self.publish_menu();

        let (updates, mut received) = mpsc::unbounded_channel();
        let view = self.view.clone();
        let mut tray = self.tray.clone();
        let repaint = self.repaint.clone();
        tokio::spawn(async move {
            // The bubble animates the running stage on its own clock.
            let mut tick = tokio::time::interval(Duration::from_millis(120));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                let update = tokio::select! {
                    update = received.recv() => match update {
                        Some(update) => update,
                        None => break,
                    },
                    _ = tick.tick() => {
                        let animate = {
                            let mut view = view.lock().expect("the view lock");
                            match view.feedback.as_mut().filter(|f| f.take == id && f.animating()) {
                                Some(feedback) => {
                                    feedback.frame += 1;
                                    true
                                }
                                None => false,
                            }
                        };
                        if animate {
                            repaint();
                        }
                        continue;
                    }
                };
                let state = {
                    let mut view = view.lock().expect("the view lock");
                    if let Some(feedback) = view.feedback.as_mut().filter(|f| f.take == id) {
                        feedback.apply(&update);
                    }
                    let state = match update {
                        Update::Level(bands) => Some(TrayState::Listening {
                            level: bands.into_iter().max().unwrap_or(0),
                        }),
                        Update::Delta(text) => {
                            view.live_transcript.push_str(&text);
                            None
                        }
                        Update::Heard(text) => {
                            view.live_transcript = text;
                            None
                        }
                        Update::Transcribing => Some(TrayState::Transcribing { frame: 0 }),
                        Update::Thinking => Some(TrayState::Thinking { frame: 0 }),
                        Update::Stage(_)
                        | Update::Progress(_)
                        | Update::StageDone { .. }
                        | Update::Answering => None,
                        Update::Output(text) => {
                            view.live_output.push_str(&text);
                            None
                        }
                    };
                    if let Some(state) = state {
                        view.tray = state;
                    }
                    state
                };
                if let (Some(state), Some(tray)) = (state, &mut tray) {
                    tray.set_state(state);
                }
                repaint();
            }
        });
        let commands = self.commands.clone();
        let task = tokio::spawn(async move {
            let trace = if live {
                pipeline::run_live(&env, start, audio_events, finished, &updates).await
            } else {
                pipeline::run_take(&env, start, audio_events, finished, &updates).await
            };
            let _ = commands.send(Command::TakeFinished(Box::new(trace)));
        });
        if let Some(active) = &mut self.active {
            active.task = Some(task);
        }
        self.repaint();
    }

    fn stop_take(&mut self) {
        crate::hold::release();
        if let Some(active) = &mut self.active {
            if let Some(capture) = active.capture.take() {
                capture.stop();
            }
            if let Some(finish) = active.finish.take() {
                let _ = finish.send(());
            }
        }
        self.view().dictating = false;
        self.set_tray(TrayState::Transcribing { frame: 0 });
        self.publish_menu();
        self.repaint();
    }

    fn cancel_take(&mut self) {
        crate::hold::release();
        self.confirmed(false);
        self.view().feedback = None;
        if let Some(mut active) = self.active.take() {
            if let Some(capture) = active.capture.take() {
                capture.stop();
            }
            if let Some(task) = active.task.take() {
                task.abort();
            }
        }
    }

    fn finished(&mut self, trace: Trace) {
        crate::hold::release();
        self.confirmed(false);
        save_trace(&trace);
        // What jevons typed during a recording is a step of it.
        if let Some(recording) = &self.recording
            && matches!(trace.delivery, Some(DeliveryOutcome::Delivered { .. }))
        {
            let session = recording.session.clone();
            let text = trace.output.clone();
            tokio::task::spawn_blocking(move || {
                if let Some(session) = session.lock().expect("the session lock").as_mut() {
                    session.delivered(&text);
                }
            });
        }
        if self.active.as_ref().is_some_and(|a| a.id == trace.take) {
            // The source may have ended by itself (a device error).
            if let Some(active) = self.active.take()
                && let Some(capture) = active.capture
            {
                capture.stop();
            }
        }
        let failed = trace.error.is_some();
        {
            let mut view = self.view();
            view.dictating = false;
            view.notice = trace.error.clone().or_else(|| match &trace.delivery {
                Some(DeliveryOutcome::OnClipboard { reason }) => {
                    Some(format!("The text is on the clipboard: {reason}"))
                }
                _ => None,
            });
            if let Some(feedback) = view.feedback.as_mut().filter(|f| f.take == trace.take) {
                feedback.finish(&trace);
                // Long enough to read the outcome, errors longer; an answer stays until Close
                // (or the next take), since reading it may take a while.
                let shown = match (failed, feedback.answer) {
                    (true, _) => Some(8),
                    (false, true) => None,
                    (false, false) => Some(4),
                };
                if let Some(seconds) = shown {
                    let commands = self.commands.clone();
                    let take = trace.take;
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_secs(seconds)).await;
                        let _ = commands.send(Command::HideFeedback(take));
                    });
                }
            }
            view.traces.push_front(trace);
            view.traces.truncate(HISTORY);
        }
        self.set_tray(if failed {
            TrayState::Error
        } else {
            self.resting()
        });
        self.publish_menu();
        self.repaint();
    }

    /// The tray state between takes.
    fn resting(&self) -> TrayState {
        if self.recording.is_some() {
            TrayState::Recording
        } else {
            TrayState::Idle
        }
    }

    /// Shows a message in the bubble for a few seconds, unless a take has it.
    fn message(&mut self, status: &str, seconds: u64) {
        if self.active.as_ref().is_some_and(|a| !a.note) {
            self.notice(status);
            return;
        }
        self.messages += 1;
        let shown = self.messages;
        self.view().feedback = Some(Feedback {
            take: MESSAGE,
            status: status.into(),
            done: true,
            message: true,
            shown,
            ..Feedback::default()
        });
        let commands = self.commands.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(seconds)).await;
            let _ = commands.send(Command::HideMessage(shown));
        });
        self.repaint();
    }

    fn record_hotkey(&self) -> Option<String> {
        self.config
            .automation
            .record_hotkey
            .clone()
            .filter(|h| !h.is_empty())
    }

    /// Starts recording what the user does.
    fn start_recording(&mut self) {
        if self.active.is_some() {
            self.notice("Finish the current take first");
            return;
        }
        let own = std::env::current_exe()
            .ok()
            .and_then(|e| e.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "jevons-desktop.exe".into());
        let session = Arc::new(Mutex::new(Some(Session::new(
            self.inspector.clone(),
            &[own.as_str()],
            Duration::from_millis(400),
        ))));
        let (events, mut observed) = mpsc::unbounded_channel();
        let handle = match self.recorder.start(events) {
            Ok(handle) => handle,
            Err(e) => {
                self.message(&format!("Cannot record: {e}"), 8);
                return;
            }
        };
        let steps = session.clone();
        let worker = std::thread::Builder::new()
            .name("recording".into())
            .spawn(move || {
                if let Some(session) = steps.lock().expect("the session lock").as_mut() {
                    session.begin();
                }
                while let Some(event) = observed.blocking_recv() {
                    if let Some(session) = steps.lock().expect("the session lock").as_mut() {
                        session.observe(event);
                    }
                }
            })
            .ok();
        self.recording = Some(ActiveRecording {
            session,
            handle: Some(handle),
            worker,
        });
        tracing::info!("Recording a demonstration");
        let how = match self.record_hotkey() {
            Some(hotkey) => format!(
                "Recording. Hold {hotkey} and say what this task is, then do it. Tap {hotkey} \
                 (or use the tray) when you are done."
            ),
            None => "Recording: do the task, then stop from the tray menu. (Set a record \
                     hotkey in the settings to say what the task is.)"
                .into(),
        };
        self.set_tray(TrayState::Recording);
        self.publish_menu();
        self.message(&how, 10);
    }

    /// Stops recording, and saves it off the agent thread.
    fn stop_recording(&mut self) {
        let Some(mut recording) = self.recording.take() else {
            return;
        };
        if let Some(handle) = recording.handle.take() {
            handle.stop();
        }
        let commands = self.commands.clone();
        let recordings = self.config.recordings_dir();
        let library = self.automations.dir().to_path_buf();
        let replacing = self.replacing.take();
        let author_with = self.runtime.connection().and_then(|c| {
            let model = self
                .config
                .automation
                .author_model
                .clone()
                .or(c.models.generative.clone())?;
            Some((c.client, model))
        });
        tokio::spawn(async move {
            let into = library.clone();
            let finished = tokio::task::spawn_blocking(move || {
                if let Some(worker) = recording.worker.take() {
                    let _ = worker.join();
                }
                let session = recording.session.lock().expect("the session lock").take();
                match session.map(Session::finish) {
                    None => Err("the recording was lost".to_string()),
                    Some(done) if done.steps.is_empty() => Err("nothing was recorded".to_string()),
                    Some(done) => bundle::save(&done, &recordings, &into)
                        .map(|dir| (done, dir))
                        .map_err(|e| e.to_string()),
                }
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()));
            let (done, dir) = match finished {
                Ok(saved) => saved,
                Err(e) => {
                    let _ = commands.send(Command::RecordingSaved(Err(e)));
                    return;
                }
            };
            let _ = commands.send(Command::RecordingSaved(Ok((
                dir.clone(),
                done.steps.len(),
                done.description.clone(),
            ))));
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let client = author_with
                .as_ref()
                .map(|(client, model)| (client, model.as_str()));
            let authored = author::author(client, &done, &name, &library, replacing.as_deref())
                .await
                .map(Box::new);
            let _ = commands.send(Command::Authored(authored));
        });
        tracing::info!("Recording stopped");
        self.set_tray(TrayState::Idle);
        self.publish_menu();
        self.message("Saving the recording…", 5);
    }

    fn recording_saved(&mut self, saved: Result<(PathBuf, usize, String), String>) {
        match saved {
            Ok((dir, steps, _)) => {
                tracing::info!(steps, dir = %dir.display(), "Recording saved");
                self.message(
                    &format!("Recorded {steps} steps. Writing the automation…"),
                    60,
                );
            }
            Err(e) => self.message(&format!("The recording was not saved: {e}"), 8),
        }
    }

    /// The author wrote an automation: ask the user to approve it.
    fn authored(&mut self, authored: Result<Box<Authored>, String>) {
        let authored = match authored {
            Ok(authored) => authored,
            Err(e) => {
                self.message(&format!("The automation was not written: {e}"), 10);
                return;
            }
        };
        tracing::info!(
            automation = %authored.automation.name,
            planned = authored.planned,
            ok = authored.report.ok(),
            "Automation written"
        );
        self.automations.reload();
        self.list_tools();
        self.publish_menu();
        self.reviewed(authored.report);
    }

    /// Checks an automation off the agent thread, then shows the result for approving.
    fn review_automation(&mut self, name: &str) {
        let library = self.automations.library();
        let Some(automation) = library.get(name).cloned() else {
            self.message(&format!("There is no automation {name}"), 5);
            return;
        };
        let commands = self.commands.clone();
        tokio::task::spawn_blocking(move || {
            let report = jevons_desktop_core::automation::check::check(&automation);
            let _ = commands.send(Command::Reviewed(Box::new(report)));
        });
        self.message(&format!("Checking {name}…"), 30);
    }

    /// Shows what an automation does and asks to approve this version, or why it cannot be.
    fn reviewed(&mut self, report: CheckReport) {
        if !report.ok() {
            let problems: Vec<String> = report
                .errors
                .iter()
                .map(|e| e.to_string())
                .chain(report.fixtures.iter().filter_map(|f| f.problem.clone()))
                .take(3)
                .collect();
            self.message(
                &format!(
                    "{} has problems, so it cannot be approved: {}. Fix it in the automations \
                     folder, or record it again.",
                    report.name,
                    problems.join("; ")
                ),
                15,
            );
            return;
        }
        let summary = &report.summary;
        let mut does: Vec<String> = summary.actions.iter().cloned().collect();
        does.extend(summary.keys.iter().map(|k| format!("press {k}")));
        if summary.types_text {
            does.push("type into the window in front".into());
        }
        let replayed: usize = report
            .fixtures
            .iter()
            .filter_map(|f| f.trace.as_ref().and_then(|t| t.replayed))
            .map(|(done, _)| done)
            .sum();
        let arguments = serde_json::json!({
            "applications": summary.apps,
            "does": does,
            "replays": format!("{replayed} recorded steps"),
            "version": report.version,
        });
        if self.confirming.is_some() {
            // A take waits on its own confirmation: leave it be, and approve later.
            self.notice(&format!(
                "{} is ready: approve it from the tray's Automations menu",
                report.name
            ));
            return;
        }
        let (reply, answer) = oneshot::channel();
        let commands = self.commands.clone();
        let name = report.name.clone();
        let version = report.version.clone();
        tokio::spawn(async move {
            let yes = answer.await.unwrap_or(false);
            let _ = commands.send(Command::ApprovalAnswered { name, version, yes });
        });
        self.confirm_requested(Confirmation {
            tool: format!("Approve {}?", report.name),
            arguments,
            reply,
        });
    }

    fn approval_answered(&mut self, name: &str, version: &str, yes: bool) {
        if !yes {
            self.message(
                &format!("{name} stays a draft: approve it later from the tray's Automations menu"),
                6,
            );
            return;
        }
        match DesktopConfig::approve(&self.config_file, name, version) {
            Ok(saved) => {
                self.config.automation.approved = saved.automation.approved;
                self.automations
                    .set_settings(self.config.automation.clone());
                self.view().config = self.config.clone();
                tracing::info!(automation = %name, "Automation approved");
                self.publish_menu();
                self.message(
                    &format!("{name} is approved: run it from the tray, a hotkey or by saying it"),
                    6,
                );
            }
            Err(e) => self.message(&format!("Cannot save the approval: {e}"), 8),
        }
    }

    /// A tray item runs an automation: at once when it takes no arguments, else after the user
    /// says them.
    fn run_automation(&mut self, name: &str, source: Option<(u32, bool)>, step_by_step: bool) {
        if self.active.is_some() || self.running.is_some() {
            self.notice("Finish the current take first");
            return;
        }
        let Some(listed) = self.automations.list().into_iter().find(|a| a.name == name) else {
            self.message(&format!("There is no automation {name}"), 5);
            return;
        };
        if !listed.approved {
            self.review_automation(name);
            return;
        }
        let takes_arguments = listed.parameters["required"]
            .as_array()
            .is_some_and(|r| !r.is_empty());
        if !takes_arguments {
            self.run_now(name, serde_json::Map::new(), step_by_step);
            return;
        }
        // Say the arguments: a take through a one-node tree that runs this automation.
        let tree = FlowTree::load(
            &jevons_desktop_core::flow::Memory::new(
                "automation",
                [(
                    "run.toml",
                    format!("automations = [{name:?}]\noutput = \"bubble\"\n").as_str(),
                )],
            ),
            &self.tools.catalog(),
        );
        if !tree.is_valid() {
            self.message(&format!("{name} cannot run: {:?}", tree.errors), 8);
            return;
        }
        let (id, held) = match source {
            Some((id, held)) => (Some(id), held),
            None => (None, false),
        };
        self.automations.step_next_run(step_by_step);
        self.begin(None, id, false, held, Some(Arc::new(tree)));
        let finish = match (id, held) {
            (Some(_), true) => "then release the hotkey",
            (Some(_), false) => "then press the hotkey again",
            (None, _) => "then choose Stop dictation in the tray",
        };
        if let Some(feedback) = self.view().feedback.as_mut() {
            feedback.status = format!(
                "Say what {name} needs ({}), {finish}",
                listed.parameters["properties"]
                    .as_object()
                    .map(|p| p.keys().cloned().collect::<Vec<_>>().join(", "))
                    .unwrap_or_default()
            );
        }
        self.repaint();
    }

    /// An automation's hotkey went down.
    fn automation_hotkey(&mut self, id: u32, name: String) {
        let held = self.config.dictation.hotkey_mode == HotkeyMode::Hold;
        match &self.active {
            Some(a) if a.source == Some(id) && !a.held && a.capture.is_some() => self.stop_take(),
            Some(_) => {}
            None => self.run_automation(&name, Some((id, held)), false),
        }
    }

    /// Runs an approved automation now, asking first when it must, with its progress in the
    /// bubble.
    fn run_now(
        &mut self,
        name: &str,
        arguments: serde_json::Map<String, serde_json::Value>,
        step_by_step: bool,
    ) {
        let host = self.automations.clone();
        let commands = self.commands.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        self.running = Some((name.to_string(), cancel.clone()));
        let asks = host.asks(name);
        let name = name.to_string();
        self.message(&format!("Running {name}…"), 300);
        self.set_tray(TrayState::Thinking { frame: 0 });
        self.publish_menu();
        tokio::spawn(async move {
            if asks {
                let (reply, answer) = oneshot::channel();
                let _ = commands.send(Command::ConfirmRequested(Confirmation {
                    tool: format!("script:{name}"),
                    arguments: serde_json::Value::Object(arguments.clone()),
                    reply,
                }));
                if !answer.await.unwrap_or(false) {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            let progress: jevons_desktop_core::automation::engine::Progress = {
                let commands = commands.clone();
                Arc::new(move |label: &str| {
                    let _ = commands.send(Command::AutomationProgress(format!("{label}…")));
                })
            };
            let trace = if cancel.load(Ordering::Relaxed) {
                let mut trace = host.run(&name, &arguments, None, cancel.clone(), false);
                if trace.error.is_none() {
                    trace.error = Some(jevons_desktop_core::automation::run::AutomationError::new(
                        jevons_desktop_core::automation::run::ErrorKind::Cancelled,
                        "not confirmed",
                    ));
                }
                trace
            } else {
                tokio::task::spawn_blocking(move || {
                    host.run(&name, &arguments, Some(progress), cancel, step_by_step)
                })
                .await
                .unwrap_or_else(|e| {
                    let mut trace = RunTrace::default_for(&e.to_string());
                    trace.error = Some(jevons_desktop_core::automation::run::AutomationError::new(
                        jevons_desktop_core::automation::run::ErrorKind::Script,
                        e.to_string(),
                    ));
                    trace
                })
            };
            let _ = commands.send(Command::AutomationFinished(Box::new(trace)));
        });
    }

    fn automation_finished(&mut self, trace: RunTrace) {
        self.running = None;
        save_run(&trace);
        let status = match &trace.error {
            None => {
                let result = trace
                    .result
                    .as_ref()
                    .filter(|r| !r.is_null())
                    .map(|r| match r {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                format!("{} done. {result}", trace.automation)
            }
            Some(error) => {
                let at = match (error.line, error.column) {
                    (Some(line), Some(column)) => format!(" (script.rhai:{line}:{column})"),
                    _ => String::new(),
                };
                format!("{} failed: {}{at}", trace.automation, error.message)
            }
        };
        let failed = trace.error.is_some();
        tracing::info!(automation = %trace.automation, ok = !failed, ms = trace.ms, "Automation finished");
        self.message(&status, if failed { 12 } else { 6 });
        self.set_tray(if failed {
            TrayState::Error
        } else {
            self.resting()
        });
        self.publish_menu();
    }

    /// The record hotkey went down: start recording (and listen for what the task is), or,
    /// while recording, listen for a note.
    fn record_pressed(&mut self, id: u32) {
        let started = self.recording.is_none();
        if started {
            self.start_recording();
            if self.recording.is_none() {
                return;
            }
        }
        if self.active.is_some() {
            return;
        }
        self.record_press = Some((id, std::time::Instant::now(), started));
        self.begin_note(id);
    }

    /// The record hotkey came up: a tap stops the recording; a hold was a note.
    fn record_released(&mut self) {
        let Some((_, since, started)) = self.record_press.take() else {
            return;
        };
        if since.elapsed() < Duration::from_millis(500) {
            if self.active.as_ref().is_some_and(|a| a.note) {
                self.cancel_take();
            }
            if !started {
                self.stop_recording();
            } else {
                self.set_tray(TrayState::Recording);
            }
        } else if self.active.as_ref().is_some_and(|a| a.note) {
            self.stop_take();
        }
    }

    /// Listens while the record hotkey is held, and transcribes what the user said for the
    /// recording.
    fn begin_note(&mut self, source: u32) {
        let Some(connection) = self.runtime.connection() else {
            return;
        };
        let (audio, audio_events) = mpsc::unbounded_channel();
        let capture = match self
            .audio
            .start(self.config.dictation.microphone.as_deref(), audio)
        {
            Ok(capture) => capture,
            Err(e) => {
                self.notice(&format!("Cannot open the microphone: {e}"));
                return;
            }
        };
        let id = self.next_take;
        self.next_take += 1;
        let (finish, finished) = oneshot::channel();
        let dictation = &self.config.dictation;
        let env = Env {
            client: connection.client,
            flows: self.flows.clone(),
            settings: pipeline::Settings {
                models: connection.models,
                realtime: connection.realtime,
                language: dictation.language.clone(),
                ..pipeline::Settings::default()
            },
            sink: None,
            investigator: None,
            reader: None,
            confirmer: None,
            tools: None,
        };
        let describing = self
            .recording
            .as_ref()
            .and_then(|r| {
                r.session
                    .lock()
                    .expect("the session lock")
                    .as_ref()
                    .map(|s| s.description().is_empty())
            })
            .unwrap_or(true);
        self.view().feedback = Some(Feedback {
            take: id,
            status: if describing {
                "Listening: say what this task is".into()
            } else {
                "Listening for a note".into()
            },
            message: true,
            ..Feedback::default()
        });
        self.active = Some(Active {
            id,
            live: false,
            source: Some(source),
            held: true,
            capture: Some(capture),
            finish: Some(finish),
            task: None,
            note: true,
        });
        if let Some(accelerator) = self.accelerators.get(&source) {
            crate::hold::hold(accelerator);
        }
        self.set_tray(TrayState::Listening { level: 0 });
        let commands = self.commands.clone();
        let task = tokio::spawn(async move {
            let (updates, _) = mpsc::unbounded_channel();
            let start = TakeStart {
                id,
                context: ContextSnapshot::default(),
                entry: None,
            };
            let said =
                pipeline::transcribe_only(&env, start, audio_events, finished, &updates).await;
            let _ = commands.send(Command::RecordingNote(said));
        });
        if let Some(active) = &mut self.active {
            active.task = Some(task);
        }
        self.repaint();
    }

    fn recording_note(&mut self, said: Result<String, String>) {
        crate::hold::release();
        if let Some(active) = self.active.take()
            && let Some(capture) = active.capture
        {
            capture.stop();
        }
        match (said, &self.recording) {
            (Ok(text), Some(recording)) => {
                let first = {
                    let mut session = recording.session.lock().expect("the session lock");
                    match session.as_mut() {
                        Some(session) => {
                            let first = session.description().is_empty();
                            session.describe(&text);
                            first
                        }
                        None => false,
                    }
                };
                self.message(
                    &if first {
                        format!("The task: {text}")
                    } else {
                        format!("Noted: {text}")
                    },
                    5,
                );
            }
            (Ok(_), None) => {}
            (Err(e), _) => self.message(&format!("Nothing was heard: {e}"), 4),
        }
        self.set_tray(self.resting());
        self.publish_menu();
    }
}

/// Writes an automation run's trace next to the take traces.
fn save_run(trace: &RunTrace) {
    let dir = jevons_desktop_core::config::user_dir().join("traces");
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let file = dir.join(format!("{millis}-automation-{}.json", trace.automation));
    if let Ok(json) = serde_json::to_vec_pretty(trace) {
        std::thread::spawn(move || {
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(file, json);
        });
    }
}

/// How many take traces `~/jevons/traces` keeps.
const SAVED_TRACES: usize = 200;

/// Writes `trace` to `~/jevons/traces` as JSON, keeping the newest [`SAVED_TRACES`].
fn save_trace(trace: &Trace) {
    let dir = jevons_desktop_core::config::user_dir().join("traces");
    let turn = trace.turn.map_or(String::new(), |t| format!("-turn{t}"));
    let file = dir.join(format!(
        "{}-take{}{turn}.json",
        trace.started_at_ms, trace.take
    ));
    let json = match serde_json::to_vec_pretty(trace) {
        Ok(json) => json,
        Err(e) => return tracing::warn!(error = %e, "Cannot serialize the trace"),
    };
    std::thread::spawn(move || {
        if let Err(e) = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&file, json)) {
            return tracing::warn!(error = %e, "Cannot save the trace");
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut traces: Vec<_> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        traces.sort();
        let excess = traces.len().saturating_sub(SAVED_TRACES);
        for old in &traces[..excess] {
            let _ = std::fs::remove_file(old);
        }
    });
}

/// Watches a folder, sending `command()` on each change in it.
fn watch(
    dir: &std::path::Path,
    commands: mpsc::UnboundedSender<Command>,
    command: impl Fn() -> Command + Send + 'static,
) -> Option<notify::RecommendedWatcher> {
    use notify::Watcher;
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if event.is_ok_and(|e| !e.kind.is_access()) {
            let _ = commands.send(command());
        }
    })
    .ok()?;
    watcher.watch(dir, notify::RecursiveMode::Recursive).ok()?;
    Some(watcher)
}

/// The automations library in the settings' folder (its guides written first), over the
/// platform's layers.
fn automation_host(
    config: &DesktopConfig,
    config_file: &std::path::Path,
    layers: &Layers,
    commands: &mpsc::UnboundedSender<Command>,
) -> Arc<AutomationHost> {
    automation_host_over(
        config,
        config_file,
        (layers.inspector.clone(), layers.actor.clone()),
        commands,
    )
}

fn automation_host_over(
    config: &DesktopConfig,
    config_file: &std::path::Path,
    (inspector, actor): (Arc<dyn ContextInspector>, Arc<dyn UiActor>),
    commands: &mpsc::UnboundedSender<Command>,
) -> Arc<AutomationHost> {
    let dir = config.automations_dir(config_file);
    match jevons_desktop_core::automation::defaults::init(&dir) {
        Ok(report) => {
            for note in report.notes {
                tracing::info!(%note, "Automations library");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, dir = %dir.display(), "Cannot prepare the automations library")
        }
    }
    let host = Arc::new(AutomationHost::new(
        &dir,
        config.automation.clone(),
        inspector,
        actor,
    ));
    // A script's confirm() asks in the bubble, like a tool call.
    let (sender, mut asked) = mpsc::unbounded_channel::<Confirmation>();
    let forward = commands.clone();
    tokio::spawn(async move {
        while let Some(confirmation) = asked.recv().await {
            let _ = forward.send(Command::ConfirmRequested(confirmation));
        }
    });
    let confirmer = Arc::new(ChannelConfirmer::new(sender));
    let runtime = tokio::runtime::Handle::current();
    host.set_confirm(Some(Arc::new(move |question: &str| {
        // Scripts run on blocking threads, which may wait on the runtime.
        runtime.block_on(confirmer.ask("confirm", &serde_json::json!({ "question": question })))
    })));
    host
}

/// Opens a folder in the platform file manager.
pub fn open_folder(dir: &std::path::Path) {
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(program).arg(dir).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_shows_phrases_as_heard_and_the_outcome_at_the_end() {
        let mut feedback = Feedback {
            take: 1,
            live: true,
            ..Feedback::default()
        };
        assert!(!feedback.apply(&Update::Level([1; 5])));
        feedback.apply(&Update::Delta("hola".into()));
        feedback.apply(&Update::Delta(" a todos".into()));
        assert_eq!(feedback.partial, "hola a todos");
        feedback.apply(&Update::Heard("Hola a todos.".into()));
        assert_eq!(
            (feedback.heard.as_str(), feedback.partial.as_str()),
            ("Hola a todos.", "")
        );
        feedback.apply(&Update::Stage(jevons_desktop_core::pipeline::Stage {
            kind: StageKind::Deciding,
            label: "what to do".into(),
            choices: vec!["ask".into(), "dictate".into()],
        }));
        assert!(feedback.animating());
        assert_eq!(feedback.status, "Deciding…");
        feedback.apply(&Update::StageDone {
            detail: "0.92".into(),
            chosen: Some("dictate".into()),
            ok: true,
        });
        feedback.apply(&Update::Stage(jevons_desktop_core::pipeline::Stage::new(
            StageKind::Investigating,
            "conversation",
        )));
        feedback.apply(&Update::Progress("reading e7".into()));
        assert_eq!(feedback.stages[1].detail, "reading e7");
        let mut trace = Trace::new(&TakeStart {
            id: 1,
            context: ContextSnapshot::default(),
            entry: None,
        });
        trace.transcript = "Hola a todos.".into();
        trace.output = "Hola a todos.".into();
        trace.delivery = Some(DeliveryOutcome::Delivered {
            method: jevons_desktop_core::platform::DeliveryMethod::Type,
        });
        feedback.finish(&trace);
        assert!(feedback.done && !feedback.failed);
        assert_eq!(feedback.status, "Inserted");
        assert_eq!(
            feedback.output, "",
            "an unchanged transcript is not shown twice"
        );
        assert_eq!(feedback.stages[0].chosen.as_deref(), Some("dictate"));
        assert_eq!(
            feedback.stages[1].ok,
            Some(true),
            "open stages close with the take"
        );
        assert!(!feedback.animating());
    }

    #[test]
    fn an_answer_stays_in_the_bubble_even_when_it_repeats_the_words() {
        let mut feedback = Feedback::default();
        let mut trace = Trace::new(&TakeStart {
            id: 1,
            context: ContextSnapshot::default(),
            entry: Some("ask".into()),
        });
        trace.transcript = "What time is it".into();
        trace.output = "What time is it".into();
        trace.delivery = Some(DeliveryOutcome::Shown);
        feedback.finish(&trace);
        assert!(feedback.answer);
        assert_eq!(feedback.status, "Answer");
        assert_eq!(feedback.output, "What time is it");
    }

    #[test]
    fn push_to_talk_feedback_keeps_the_streamed_words_as_the_transcript() {
        let mut feedback = Feedback::default();
        feedback.apply(&Update::Delta("hello".into()));
        feedback.apply(&Update::Thinking);
        assert_eq!(feedback.heard, "hello");
        assert_eq!(feedback.status, "Thinking…");
    }
}
