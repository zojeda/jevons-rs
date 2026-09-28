//! The agent: hotkeys, context capture, microphone takes and the pipeline, with the
//! tray and the inspector as views. It runs on its own thread; takes run on the runtime's
//! workers so a slow generation never blocks the hotkey.

use crate::runtime::{Runtime, Status};
use crate::tray::Tray;
use jevons_desktop_core::config::DesktopConfig;
use jevons_desktop_core::context::ContextSnapshot;
use jevons_desktop_core::icons::TrayState;
use jevons_desktop_core::pipeline::{self, Env, TakeStart, Trace, Update};
use jevons_desktop_core::platform::{
    AudioDevice, AudioSource, Binding, CaptureHandle, ContextProvider, HotkeyAction, HotkeyEvent,
    MenuCommand, MenuModel, TextSink, TrayBackend,
};
use jevons_desktop_core::profile::{Profiles, Resolution};
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
        actions: Vec<(u32, HotkeyAction)>,
        errors: Vec<String>,
    },
    Menu(MenuCommand),
    /// Save and apply new settings.
    Apply(Box<DesktopConfig>),
    /// Read the context now, for the inspector.
    RefreshContext,
    /// Read the context after a delay, so the user can switch to the target application.
    CaptureContextIn(Duration),
    ReloadProfiles,
    RuntimeChanged,
    /// Apply the settings to the runtime again, such as after a model finished downloading.
    ReloadRuntime,
    TakeFinished(Box<Trace>),
    /// A live dictation turn was delivered; the take goes on.
    TurnFinished(Box<Trace>),
    /// Live dictation ended, with the reason when it failed.
    LiveEnded {
        take: u64,
        error: Option<String>,
    },
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
    pub resolution: Option<Resolution>,
    pub traces: VecDeque<Trace>,
    pub profiles: Arc<Profiles>,
    pub forced_profile: Option<String>,
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
    pub quit: bool,
}

pub type SharedView = Arc<Mutex<View>>;

struct Active {
    id: u64,
    /// Live dictation rather than push-to-talk.
    live: bool,
    /// The hotkey that started it; its release ends a push-to-talk take.
    source: Option<u32>,
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
    sink: Arc<Mutex<Box<dyn TextSink>>>,
    audio: Box<dyn AudioSource>,
    profiles: Arc<Profiles>,
    hotkeys: std::collections::HashMap<u32, HotkeyAction>,
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
        let profiles_dir = config.profiles_dir(&config_file);
        let _ = std::fs::create_dir_all(&profiles_dir);
        let watcher = watch(&profiles_dir, commands.clone());
        let mut agent = Self {
            profiles: Arc::new(Profiles::load_dir(&profiles_dir)),
            config,
            config_file,
            runtime,
            tray: layers.tray,
            context: layers.context,
            sink: Arc::new(Mutex::new(layers.sink)),
            audio: layers.audio,
            hotkeys: std::collections::HashMap::new(),
            active: None,
            next_take: 1,
            view,
            repaint,
            commands,
            _watcher: watcher,
        };
        {
            let mut view = agent.view.lock().expect("the view lock");
            view.profiles = agent.profiles.clone();
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
        agent
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
        let (forced, paused) = {
            let view = self.view();
            (view.forced_profile.clone(), view.context_paused)
        };
        let live = self.active.as_ref().is_some_and(|a| a.live);
        let menu = MenuModel {
            busy: self.active.is_some(),
            dictating: self.active.as_ref().is_some_and(|a| !a.live),
            live,
            profiles: self
                .profiles
                .iter()
                .map(|p| (p.spec.id.clone(), p.display_name().to_string()))
                .collect(),
            forced,
            context_paused: paused,
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
                Some(HotkeyAction::Dictate { profile }) => {
                    // Key repeat and other keys while a take runs change nothing.
                    if self.active.is_none() {
                        let profile = profile.clone();
                        self.begin(profile, Some(id), false);
                    }
                }
                Some(HotkeyAction::LiveDictation) => self.toggle_live(),
                None => {}
            },
            Command::Hotkey(HotkeyEvent::Released(id)) => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| !a.live && a.source == Some(id) && a.capture.is_some())
                {
                    self.stop_take();
                }
            }
            Command::HotkeysRegistered { actions, errors } => {
                self.hotkeys = actions.into_iter().collect();
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
            Command::ReloadProfiles => self.reload_profiles(),
            Command::ReloadRuntime => self.runtime.apply(&self.config, &self.config_file),
            Command::RuntimeChanged => {
                let status = self.runtime.status();
                if self.active.is_none() {
                    let state = match status {
                        Status::Ready { .. } | Status::Remote { .. } => TrayState::Idle,
                        Status::Failed(_) => TrayState::Error,
                        _ => TrayState::Offline,
                    };
                    self.set_tray(state);
                }
                self.view().runtime = Some(status);
                self.repaint();
            }
            Command::TakeFinished(trace) => self.finished(*trace),
            Command::TurnFinished(trace) => self.turn_finished(*trace),
            Command::LiveEnded { take, error } => self.live_ended(take, error),
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
            MenuCommand::ForceProfile(profile) => {
                self.view().forced_profile = profile;
                self.publish_menu();
                self.refresh_context(true);
            }
            MenuCommand::ShowInspector => {
                self.view().show_window = true;
                self.repaint();
            }
            MenuCommand::ToggleContextPause => {
                {
                    let mut view = self.view();
                    view.context_paused = !view.context_paused;
                }
                self.publish_menu();
                self.repaint();
            }
            MenuCommand::ReloadProfiles => self.reload_profiles(),
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

    /// Starts or stops push-to-talk dictation from the menu.
    fn toggle(&mut self) {
        match &self.active {
            Some(active) if !active.live && active.capture.is_some() => self.stop_take(),
            Some(active) if active.live => self.notice("Live dictation is running"),
            Some(_) => self.notice("The last take is still being processed"),
            None => self.begin(None, None, false),
        }
    }

    fn toggle_live(&mut self) {
        match &self.active {
            Some(active) if active.live && active.capture.is_some() => self.stop_take(),
            Some(active) if active.live => {}
            Some(_) => self.notice("Finish the current take first"),
            None => self.begin(None, None, true),
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
        let forced = self.view().forced_profile.clone();
        let mut view = self.view();
        match snapshot {
            Ok(snapshot) => {
                view.resolution = Some(self.profiles.resolve(&snapshot, forced.as_deref()));
                view.context = Some(snapshot);
                view.context_error = None;
            }
            Err(e) if !e.is_empty() => view.context_error = Some(e),
            Err(_) => {}
        }
        drop(view);
        self.repaint();
    }

    fn reload_profiles(&mut self) {
        let dir = self.config.profiles_dir(&self.config_file);
        self.profiles = Arc::new(Profiles::load_dir(&dir));
        self.view().profiles = self.profiles.clone();
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
        let profiles_changed =
            config.profiles_dir(&self.config_file) != self.config.profiles_dir(&self.config_file);
        self.config = config;
        if hotkeys_changed && let Some(tray) = &self.tray {
            tray.hotkeys(Binding::from_settings(&self.config.dictation));
        }
        if profiles_changed {
            let dir = self.config.profiles_dir(&self.config_file);
            let _ = std::fs::create_dir_all(&dir);
            self._watcher = watch(&dir, self.commands.clone());
            self.reload_profiles();
        }
        self.runtime.apply(&self.config, &self.config_file);
        self.view().config = self.config.clone();
        self.repaint();
    }

    /// Starts a take: push-to-talk, or live dictation when `live`. `source` is the hotkey
    /// whose release ends a push-to-talk take.
    fn begin(&mut self, profile: Option<String>, source: Option<u32>, live: bool) {
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
        let env = Env {
            client: connection.client,
            profiles: self.profiles.clone(),
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
            sink: Some(self.sink.clone()),
        };
        let start = TakeStart {
            id,
            context: context.clone(),
            forced_profile: profile.or_else(|| self.view().forced_profile.clone()),
        };
        {
            let mut view = self.view();
            view.dictating = true;
            view.live_transcript.clear();
            view.live_output.clear();
            view.notice = None;
            view.resolution = Some(
                self.profiles
                    .resolve(&context, start.forced_profile.as_deref()),
            );
            view.context = Some(context);
        }
        self.active = Some(Active {
            id,
            live,
            source,
            capture: Some(capture),
            finish: Some(finish),
            task: None,
        });
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
                    let state = match update {
                        Update::Level(bands) => Some(TrayState::Listening {
                            level: bands.into_iter().max().unwrap_or(0),
                        }),
                        Update::Delta(text) => {
                            view.live_transcript.push_str(&text);
                            None
                        }
                        Update::Transcribing => Some(TrayState::Transcribing { frame: 0 }),
                        Update::Thinking => Some(TrayState::Thinking { frame: 0 }),
                        Update::Listening => {
                            view.live_transcript.clear();
                            view.live_output.clear();
                            Some(TrayState::Listening { level: 0 })
                        }
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
        let task = if live {
            tokio::spawn(async move {
                let turns = commands.clone();
                let result =
                    pipeline::run_live(&env, start, audio_events, finished, &updates, |trace| {
                        let _ = turns.send(Command::TurnFinished(Box::new(trace)));
                    })
                    .await;
                let _ = commands.send(Command::LiveEnded {
                    take: id,
                    error: result.err(),
                });
            })
        } else {
            tokio::spawn(async move {
                let trace = pipeline::run_take(&env, start, audio_events, finished, &updates).await;
                let _ = commands.send(Command::TakeFinished(Box::new(trace)));
            })
        };
        if let Some(active) = &mut self.active {
            active.task = Some(task);
        }
        self.repaint();
    }

    fn stop_take(&mut self) {
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
        if let Some(mut active) = self.active.take() {
            if let Some(capture) = active.capture.take() {
                capture.stop();
            }
            if let Some(task) = active.task.take() {
                task.abort();
            }
        }
    }

    fn turn_finished(&mut self, trace: Trace) {
        save_trace(&trace);
        let mut view = self.view();
        view.notice = trace.error.clone();
        view.traces.push_front(trace);
        view.traces.truncate(HISTORY);
        drop(view);
        self.repaint();
    }

    fn live_ended(&mut self, take: u64, error: Option<String>) {
        if self.active.as_ref().is_some_and(|a| a.id == take)
            && let Some(active) = self.active.take()
            && let Some(capture) = active.capture
        {
            capture.stop();
        }
        {
            let mut view = self.view();
            view.dictating = false;
            if error.is_some() {
                view.notice = error.clone();
            }
        }
        self.set_tray(if error.is_some() {
            TrayState::Error
        } else {
            TrayState::Idle
        });
        self.publish_menu();
        self.repaint();
    }

    fn finished(&mut self, trace: Trace) {
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
                Some(jevons_desktop_core::platform::DeliveryOutcome::OnClipboard { reason }) => {
                    Some(format!("The text is on the clipboard: {reason}"))
                }
                _ => None,
            });
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
            let _ = commands.send(Command::ReloadProfiles);
        }
    })
    .ok()?;
    watcher
        .watch(dir, notify::RecursiveMode::NonRecursive)
        .ok()?;
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
