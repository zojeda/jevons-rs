//! The agent: hotkeys, context capture, microphone takes and the pipeline, with the
//! tray and the inspector as views. It runs on its own thread; takes run on the runtime's
//! workers so a slow generation never blocks the hotkey.

use crate::runtime::{Runtime, Status};
use crate::tray::Tray;
use jevons_desktop_core::config::{DesktopConfig, HotkeyMode};
use jevons_desktop_core::context::ContextSnapshot;
use jevons_desktop_core::flow::confirm::{ChannelConfirmer, Confirmation};
use jevons_desktop_core::flow::investigate::Investigate;
use jevons_desktop_core::flow::investigator::{Investigator, PathCache};
use jevons_desktop_core::flow::tools::ToolHost;
use jevons_desktop_core::flow::walk::{self, FlowStep};
use jevons_desktop_core::flow::{Catalog, FlowError, FlowTree, defaults};
use jevons_desktop_core::icons::TrayState;
use jevons_desktop_core::pipeline::{self, Env, TakeStart, Trace, Update};
use jevons_desktop_core::platform::{
    AudioDevice, AudioSource, Binding, CaptureHandle, ContextInspector, ContextProvider,
    DeliveryOutcome, HotkeyAction, HotkeyEvent, MenuCommand, MenuModel, TextSink, TrayBackend,
};
use jevons_desktop_core::recorded::RecordedTree;
use std::collections::VecDeque;
use std::path::PathBuf;
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
    ReloadFlows,
    RuntimeChanged,
    /// The MCP servers listed their tools: check the flows against them.
    ToolsListed(Vec<String>),
    /// Apply the settings to the runtime again, such as after a model finished downloading.
    ReloadRuntime,
    TakeFinished(Box<Trace>),
    /// Hides the feedback bubble of a finished take, unless a newer take shows it.
    HideFeedback(u64),
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
    /// The route through the flow tree and the other steps taken.
    pub steps: Vec<String>,
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
            Update::Transcribing => self.status = "Transcribing…".into(),
            Update::Thinking => {
                self.status = "Thinking…".into();
                if !self.live && self.heard.is_empty() {
                    self.heard = std::mem::take(&mut self.partial);
                }
            }
            Update::Step(step) => self.steps.push(step.clone()),
            Update::Output(text) => self.output.push_str(text),
            Update::Answering => {
                self.answer = true;
                self.output.clear();
            }
        }
        true
    }

    /// The take's outcome.
    fn finish(&mut self, trace: &Trace) {
        self.done = true;
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
    pub quit: bool,
}

pub type SharedView = Arc<Mutex<View>>;

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
}

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
    /// The tools the settings register.
    tools: Arc<ToolHost>,
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
}

/// The platform layers the agent drives.
pub struct Layers {
    pub context: Box<dyn ContextProvider>,
    pub inspector: Arc<dyn ContextInspector>,
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
        let flows_dir = config.flows_dir(&config_file);
        let tools = Arc::new(ToolHost::new(&config.tools, &config.mcp));
        let (tree, notes) = defaults::open(&flows_dir, &tools.catalog());
        let errors = tree.errors.clone();
        let flows = if tree.is_valid() {
            tree
        } else {
            // Nothing good to keep yet: the built-in tree runs until the folder is fixed.
            FlowTree::load(&defaults::builtin(), &Catalog::default())
        };
        let watcher = watch(&flows_dir, commands.clone());
        let mut agent = Self {
            tools,
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
            tray.hotkeys(Binding::from_settings(&agent.config.dictation));
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
            busy: self.active.is_some(),
            dictating: self.active.as_ref().is_some_and(|a| !a.live),
            live,
            entries: self.flows.entries(),
            start,
            context_paused: paused,
            feedback: self.config.dictation.live_feedback,
        };
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
                None => {}
            },
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
            Command::ToolsListed(problems) => {
                self.view().tool_problems = problems;
                self.reload_flows();
            }
            Command::ReloadFlows => self.reload_flows(),
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
            None => self.begin(entry, Some(id), live, mode == HotkeyMode::Hold),
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
            None => self.begin(None, None, false, false),
        }
    }

    fn toggle_live(&mut self) {
        match &self.active {
            Some(active) if active.live && active.capture.is_some() => self.stop_take(),
            Some(active) if active.live => {}
            Some(_) => self.notice("Finish the current take first"),
            None => self.begin(None, None, true, false),
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
            if let Some(feedback) = view.feedback.as_mut() {
                feedback.confirm = Some(PendingCall {
                    tool: confirmation.tool,
                    arguments,
                });
                feedback.status = "Run this tool? Enter runs it, Esc cancels".into();
            }
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
            != Binding::from_settings(&self.config.dictation);
        let flows_changed =
            config.flows_dir(&self.config_file) != self.config.flows_dir(&self.config_file);
        let tools_changed = config.tools != self.config.tools || config.mcp != self.config.mcp;
        self.config = config;
        if hotkeys_changed && let Some(tray) = &self.tray {
            tray.hotkeys(self.bindings());
        }
        if flows_changed {
            let dir = self.config.flows_dir(&self.config_file);
            let _ = std::fs::create_dir_all(&dir);
            self._watcher = watch(&dir, self.commands.clone());
            self.reload_flows();
        }
        if tools_changed || flows_changed {
            self.tools = Arc::new(ToolHost::new(&self.config.tools, &self.config.mcp));
            self.list_tools();
        }
        self.runtime.apply(&self.config, &self.config_file);
        self.view().config = self.config.clone();
        self.repaint();
    }

    /// Starts a take: push-to-talk, or live dictation when `live`. `source` is the hotkey that
    /// started it; when `held`, its release ends the take.
    fn begin(&mut self, entry: Option<String>, source: Option<u32>, live: bool, held: bool) {
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
            flows: self.flows.clone(),
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
            while let Some(update) = received.recv().await {
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
                        Update::Step(_) | Update::Answering => None,
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
                // Long enough to read the outcome; errors and answers stay longer.
                let words = trace.output.split_whitespace().count() as u64;
                let shown = Duration::from_secs(match (failed, feedback.answer) {
                    (true, _) => 8,
                    (false, true) => (6 + words / 3).min(60),
                    (false, false) => 4,
                });
                let commands = self.commands.clone();
                let take = trace.take;
                tokio::spawn(async move {
                    tokio::time::sleep(shown).await;
                    let _ = commands.send(Command::HideFeedback(take));
                });
            }
            view.traces.push_front(trace);
            view.traces.truncate(HISTORY);
        }
        self.set_tray(if failed {
            TrayState::Error
        } else {
            TrayState::Idle
        });
        self.publish_menu();
        self.repaint();
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

fn watch(
    dir: &std::path::Path,
    commands: mpsc::UnboundedSender<Command>,
) -> Option<notify::RecommendedWatcher> {
    use notify::Watcher;
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if event.is_ok_and(|e| !e.kind.is_access()) {
            let _ = commands.send(Command::ReloadFlows);
        }
    })
    .ok()?;
    watcher.watch(dir, notify::RecursiveMode::Recursive).ok()?;
    Some(watcher)
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
        feedback.apply(&Update::Step("Route dictate".into()));
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
        assert_eq!(feedback.steps, ["Route dictate"]);
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
