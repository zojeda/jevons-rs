//! The tray icon, its menu and the global hotkey, on their own thread with a tao event loop
//! (the dioxus-native window keeps the main thread). The icon animates here: the waveform follows the
//! microphone level the agent reports, and the processing dots advance on a timer.

use crate::agent::Command;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use jevons_desktop_core::history::History;
use jevons_desktop_core::icons::{self, FRAME, TrayState};
use jevons_desktop_core::platform::{
    Binding, HotkeyAction, HotkeyEvent, MachineEntry, MenuCommand, MenuModel, TrayBackend,
};
use std::time::Instant;
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::platform::run_return::EventLoopExtRunReturn;
use tokio::sync::mpsc::UnboundedSender;
use tray_icon::menu::{
    CheckMenuItem, IsMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu,
};
use tray_icon::{Icon, MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};

/// Messages to the tray thread.
#[derive(Debug)]
pub enum TrayMessage {
    State(TrayState),
    Menu(MenuModel),
    /// Registers these hotkeys, replacing the previous ones.
    Hotkeys(Vec<Binding>),
    /// Kernel autotuning started; the icon turns amber while it lasts.
    Tuning,
    Quit,
}

/// The tray icon's screen rectangle in physical pixels (x, y, width, height), for placing the
/// feedback bubble next to it.
static ICON_RECT: std::sync::Mutex<Option<(f64, f64, u32, u32)>> = std::sync::Mutex::new(None);

pub fn icon_rect() -> Option<(f64, f64, u32, u32)> {
    *ICON_RECT.lock().expect("the tray rectangle lock")
}

fn remember_rect(icon: &TrayIcon) {
    if let Some(rect) = icon.rect() {
        *ICON_RECT.lock().expect("the tray rectangle lock") = Some((
            rect.position.x,
            rect.position.y,
            rect.size.width,
            rect.size.height,
        ));
    }
}

/// Talks to the tray thread.
#[derive(Clone)]
pub struct Tray {
    proxy: EventLoopProxy<TrayMessage>,
}

impl Tray {
    pub fn hotkeys(&self, bindings: Vec<Binding>) {
        let _ = self.proxy.send_event(TrayMessage::Hotkeys(bindings));
    }

    pub fn quit(&self) {
        let _ = self.proxy.send_event(TrayMessage::Quit);
    }
}

impl TrayBackend for Tray {
    fn set_state(&mut self, state: TrayState) {
        let _ = self.proxy.send_event(TrayMessage::State(state));
    }

    fn set_menu(&mut self, menu: &MenuModel) {
        let _ = self.proxy.send_event(TrayMessage::Menu(menu.clone()));
    }
}

/// Starts the tray thread. Menu clicks, tray clicks and hotkey presses go to `commands`.
pub fn spawn(commands: UnboundedSender<Command>) -> Result<Tray, String> {
    let (ready, started) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("tray".into())
        .spawn(move || run(commands, ready))
        .map_err(|e| e.to_string())?;
    started
        .recv()
        .map_err(|_| "the tray thread ended".to_string())?
}

fn run(commands: UnboundedSender<Command>, ready: std::sync::mpsc::Sender<Result<Tray, String>>) {
    let mut builder = EventLoopBuilder::<TrayMessage>::with_user_event();
    #[cfg(windows)]
    tao::platform::windows::EventLoopBuilderExtWindows::with_any_thread(&mut builder, true);
    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    tao::platform::unix::EventLoopBuilderExtUnix::with_any_thread(&mut builder, true);
    #[cfg(target_os = "macos")]
    {
        let _ = ready.send(Err("the macOS tray is not available yet".into()));
        return;
    }
    #[allow(unreachable_code)]
    let mut event_loop = builder.build();
    let _ = ready.send(Ok(Tray {
        proxy: event_loop.create_proxy(),
    }));
    let tuning = std::sync::Mutex::new(event_loop.create_proxy());
    crate::tuning::on_start(move || {
        let _ = tuning
            .lock()
            .expect("the tray proxy lock")
            .send_event(TrayMessage::Tuning);
    });

    let hotkeys = match GlobalHotKeyManager::new() {
        Ok(manager) => Some(manager),
        Err(e) => {
            tracing::warn!(error = %e, "Global hotkeys are unavailable");
            None
        }
    };
    let events = commands.clone();
    GlobalHotKeyEvent::set_event_handler(Some(move |event: GlobalHotKeyEvent| {
        let event = match event.state {
            HotKeyState::Pressed => HotkeyEvent::Pressed(event.id),
            HotKeyState::Released => HotkeyEvent::Released(event.id),
        };
        let _ = events.send(Command::Hotkey(event));
    }));
    let events = commands.clone();
    MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
        if let Some(command) = menu_command(event.id.as_ref()) {
            let _ = events.send(Command::Menu(command));
        }
    }));
    let events = commands.clone();
    TrayIconEvent::set_event_handler(Some(move |event: TrayIconEvent| {
        if let TrayIconEvent::Click {
            button: MouseButton::Left,
            button_state: MouseButtonState::Up,
            ..
        } = event
        {
            let _ = events.send(Command::Menu(MenuCommand::ToggleDictation));
        }
    }));

    let frames: Vec<Icon> = icons::frames()
        .into_iter()
        .filter_map(|(_, pixels)| Icon::from_rgba(pixels, icons::SIZE, icons::SIZE).ok())
        .collect();
    let mut tray: Option<TrayIcon> = None;
    // The runtime starts loading as soon as the app starts.
    let mut state = TrayState::Loading;
    let mut shown: Option<TrayState> = None;
    let mut menu = MenuModel::default();
    let mut registered: Vec<HotKey> = Vec::new();
    let mut next_frame: Option<Instant> = None;

    event_loop.run_return(move |event, _, control_flow| {
        match event {
            Event::NewEvents(StartCause::Init) => {
                match TrayIconBuilder::new()
                    .with_menu(Box::new(build_menu(&menu)))
                    .with_menu_on_left_click(false)
                    .with_tooltip(state.tooltip())
                    .with_icon(frames[icons::frame_index(state)].clone())
                    .build()
                {
                    Ok(icon) => {
                        shown = Some(state);
                        remember_rect(&icon);
                        tray = Some(icon);
                    }
                    Err(e) => tracing::error!(error = %e, "Cannot create the tray icon"),
                }
            }
            Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                state = state.advance();
            }
            Event::UserEvent(TrayMessage::State(new)) => {
                // Keep the animation phase when only the level or phase is re-sent.
                if std::mem::discriminant(&new) != std::mem::discriminant(&state)
                    || matches!(new, TrayState::Listening { .. })
                {
                    // The taskbar may have moved the icon since.
                    if !matches!(state, TrayState::Listening { .. })
                        && let Some(icon) = &tray
                    {
                        remember_rect(icon);
                    }
                    state = new;
                }
            }
            // The machines' moves publish the menu often: it is built again only when it
            // differs.
            Event::UserEvent(TrayMessage::Menu(model)) if model != menu => {
                menu = model;
                if let Some(tray) = &tray {
                    tray.set_menu(Some(Box::new(build_menu(&menu))));
                }
            }
            Event::UserEvent(TrayMessage::Menu(_)) => {}
            Event::UserEvent(TrayMessage::Hotkeys(bindings)) => {
                let Some(manager) = &hotkeys else {
                    let _ = commands.send(Command::HotkeysRegistered {
                        actions: Vec::new(),
                        errors: vec!["Global hotkeys are unavailable".into()],
                    });
                    return;
                };
                let _ = manager.unregister_all(&registered);
                registered.clear();
                let mut actions: Vec<(u32, HotkeyAction, String)> = Vec::new();
                let mut errors = Vec::new();
                for binding in bindings {
                    match binding.accelerator.parse::<HotKey>() {
                        Ok(key) if actions.iter().any(|(id, _, _)| *id == key.id()) => {
                            errors.push(format!("{} is assigned twice", binding.accelerator));
                        }
                        Ok(key) => match manager.register(key) {
                            Ok(()) => {
                                registered.push(key);
                                actions.push((key.id(), binding.action, binding.accelerator));
                            }
                            Err(e) => {
                                errors
                                    .push(format!("Cannot register {}: {e}", binding.accelerator));
                            }
                        },
                        Err(e) => {
                            errors.push(format!("Invalid hotkey {:?}: {e}", binding.accelerator))
                        }
                    }
                }
                let _ = commands.send(Command::HotkeysRegistered { actions, errors });
            }
            Event::UserEvent(TrayMessage::Tuning) => {}
            Event::UserEvent(TrayMessage::Quit) => {
                tray = None;
                *control_flow = ControlFlow::Exit;
                return;
            }
            _ => {}
        }
        // Tuning outranks every state but listening, which must stay visible while speaking.
        let tuning = crate::tuning::active();
        let display = if tuning && !matches!(state, TrayState::Listening { .. }) {
            TrayState::Tuning
        } else {
            state
        };
        if let Some(icon) = &tray
            && shown != Some(display)
        {
            let _ = icon.set_icon(Some(frames[icons::frame_index(display)].clone()));
            if shown.is_none_or(|s| std::mem::discriminant(&s) != std::mem::discriminant(&display))
            {
                let _ = icon.set_tooltip(Some(display.tooltip()));
            }
            shown = Some(display);
        }
        if tuning {
            // Check again soon to notice when tuning ends.
            *control_flow =
                ControlFlow::WaitUntil(Instant::now() + std::time::Duration::from_millis(500));
        } else if state.animates() {
            let now = Instant::now();
            let due = next_frame.filter(|t| *t > now).unwrap_or(now + FRAME);
            next_frame = Some(due);
            *control_flow = ControlFlow::WaitUntil(due);
        } else {
            next_frame = None;
            *control_flow = ControlFlow::Wait;
        }
    });
}

/// What the menu shows in place of the recording and automation items while automations
/// are off.
const AUTOMATIONS_SOON: &str = "Automations (coming soon)";

/// A machine's line in the menu: what it is called and the state it is in.
fn machine_line(entry: &MachineEntry) -> String {
    format!("{} · {}", entry.label, entry.state)
}

/// The Machines submenu's title: how many tasks run shows without opening it.
fn machines_title(machines: &[MachineEntry]) -> String {
    match machines.iter().filter(|m| m.task).count() {
        0 => "Machines".into(),
        1 => "Machines · 1 task runs".into(),
        tasks => format!("Machines · {tasks} tasks run"),
    }
}

/// What runs: the root and each agent, a click on which shows it in the Machines tab, and
/// under each agent its tasks, each with what it waits for and what to do with it.
fn machines_menu(machines: &[MachineEntry]) -> Submenu {
    let menu = Submenu::with_id("machines", machines_title(machines), true);
    for entry in machines {
        let line = machine_line(entry);
        if !entry.task {
            let _ = menu.append(&MenuItem::with_id(
                format!("machine:{}", entry.id),
                line,
                true,
                None,
            ));
            continue;
        }
        let task = Submenu::with_id(format!("task:{}", entry.id), format!("    {line}"), true);
        let waits = if entry.waiting.is_empty() {
            "Working".to_string()
        } else {
            format!("Waits for {}", entry.waiting.join(", "))
        };
        let _ = task.append_items(&[
            &MenuItem::with_id(format!("waits:{}", entry.id), waits, false, None),
            &MenuItem::with_id(
                format!("machine:{}", entry.id),
                "Show in the Machines tab",
                true,
                None,
            ),
            &MenuItem::with_id(format!("end:{}", entry.id), "Cancel", true, None),
        ]);
        let _ = menu.append(&task);
    }
    if machines.is_empty() {
        let _ = menu.append(&MenuItem::with_id(
            "machines-none",
            "Nothing runs yet",
            false,
            None,
        ));
    }
    menu
}

fn build_menu(model: &MenuModel) -> Menu {
    let menu = Menu::new();
    let dictation = if model.dictating {
        "Stop dictation"
    } else {
        "Start dictation"
    };
    let start = Submenu::with_id("start", "Start takes at", true);
    let _ = start.append(&CheckMenuItem::with_id(
        "start:",
        "The root (automatic)",
        true,
        model.start.is_none(),
        None,
    ));
    let _ = start.append(&PredefinedMenuItem::separator());
    for (path, _) in &model.entries {
        let _ = start.append(&CheckMenuItem::with_id(
            format!("start:{path}"),
            path,
            true,
            model.start.as_ref() == Some(path),
            None,
        ));
    }
    let automations = Submenu::with_id("automations", "Automations", true);
    for (name, description, approved) in &model.automations {
        let entry = Submenu::with_id(format!("automation:{name}"), name, true);
        let _ = entry.append(&MenuItem::with_id(
            format!("about:{name}"),
            description,
            false,
            None,
        ));
        if *approved {
            let _ = entry.append(&MenuItem::with_id(
                format!("run:{name}"),
                "Run",
                !model.busy,
                None,
            ));
            let _ = entry.append(&MenuItem::with_id(
                format!("step:{name}"),
                "Run step by step",
                !model.busy,
                None,
            ));
        } else {
            let _ = entry.append(&MenuItem::with_id(
                format!("approve:{name}"),
                "Review and approve…",
                !model.busy,
                None,
            ));
        }
        let _ = entry.append(&MenuItem::with_id(
            format!("again:{name}"),
            "Record it again…",
            !model.busy,
            None,
        ));
        let _ = automations.append(&entry);
    }
    if model.automations.is_empty() {
        let _ = automations.append(&MenuItem::with_id(
            "none",
            "None yet: record one",
            false,
            None,
        ));
    }
    let _ = automations.append(&PredefinedMenuItem::separator());
    let _ = automations.append(&MenuItem::with_id(
        "automations-folder",
        "Open the automations folder",
        true,
        None,
    ));
    // Not while a take, an automation or a recording uses the folders.
    let idle = !model.busy && !model.recording;
    let clear = Submenu::with_id("clear", "Clear history", idle);
    let _ = clear.append_items(&[
        &MenuItem::with_id("clear:logs", "Logs…", true, None),
        &MenuItem::with_id("clear:traces", "Take traces…", true, None),
        &MenuItem::with_id("clear:trees", "Recorded interfaces…", true, None),
        &MenuItem::with_id("clear:recordings", "Recordings…", true, None),
        &MenuItem::with_id("clear:machines", "Tasks that run…", true, None),
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id("clear:all", "All of it…", true, None),
    ]);
    let live = if model.live {
        "Stop live dictation"
    } else {
        "Start live dictation"
    };
    let _ = menu.append_items(&[
        &MenuItem::with_id("toggle", dictation, !model.live, None),
        &MenuItem::with_id("live", live, !model.dictating, None),
        &MenuItem::with_id("cancel", "Cancel the current take", model.busy, None),
        &MenuItem::with_id(
            "cancel:tasks",
            "Cancel the tasks that run",
            model.tasks,
            None,
        ),
        &MenuItem::with_id(
            "conversation",
            "Show the task's conversation",
            model.conversation,
            None,
        ),
        &machines_menu(&model.machines),
        &start,
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id("inspector", "Show context inspector", true, None),
        &CheckMenuItem::with_id("feedback", "Live feedback", true, model.feedback, None),
        &CheckMenuItem::with_id(
            "pause",
            "Pause context capture",
            true,
            model.context_paused,
            None,
        ),
        &PredefinedMenuItem::separator(),
    ]);
    // Automations are coming soon: until the settings turn them on, one line says so.
    let record = MenuItem::with_id(
        "record",
        if model.recording {
            "Stop recording"
        } else {
            "Record an automation…"
        },
        !model.busy || model.recording,
        None,
    );
    let discard = MenuItem::with_id(
        "record:discard",
        "Discard the recording…",
        model.recording,
        None,
    );
    let soon = MenuItem::with_id("automations-soon", AUTOMATIONS_SOON, false, None);
    let items: &[&dyn IsMenuItem] = if model.automations_on {
        &[&record, &discard, &automations]
    } else {
        &[&soon]
    };
    let _ = menu.append_items(items);
    let _ = menu.append_items(&[
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id("reload", "Reload the flow tree", true, None),
        &MenuItem::with_id("config", "Open settings folder", true, None),
        &MenuItem::with_id(
            "reset-settings",
            "Reset settings to the defaults…",
            idle,
            None,
        ),
        &MenuItem::with_id("logs", "Open logs and traces", true, None),
        &clear,
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id("quit", "Quit", true, None),
    ]);
    menu
}

fn menu_command(id: &str) -> Option<MenuCommand> {
    Some(match id {
        "toggle" => MenuCommand::ToggleDictation,
        "record" => MenuCommand::ToggleRecording,
        "record:discard" => MenuCommand::DiscardRecording,
        "automations-folder" => MenuCommand::OpenAutomationsFolder,
        run if run.starts_with("run:") => MenuCommand::RunAutomation(run[4..].to_string()),
        step if step.starts_with("step:") => MenuCommand::RunStepByStep(step[5..].to_string()),
        again if again.starts_with("again:") => MenuCommand::RecordAgain(again[6..].to_string()),
        approve if approve.starts_with("approve:") => {
            MenuCommand::ApproveAutomation(approve[8..].to_string())
        }
        "live" => MenuCommand::ToggleLiveDictation,
        "cancel" => MenuCommand::CancelTake,
        "cancel:tasks" => MenuCommand::CancelTasks,
        "conversation" => MenuCommand::ShowConversation,
        machine if machine.starts_with("machine:") => {
            MenuCommand::ShowMachine(machine[8..].parse().ok()?)
        }
        end if end.starts_with("end:") => MenuCommand::CancelOneTask(end[4..].parse().ok()?),
        "logs" => MenuCommand::OpenLogsFolder,
        "inspector" => MenuCommand::ShowInspector,
        "pause" => MenuCommand::ToggleContextPause,
        "feedback" => MenuCommand::ToggleFeedback,
        "reload" => MenuCommand::ReloadFlows,
        "config" => MenuCommand::OpenConfigFolder,
        "reset-settings" => MenuCommand::ResetSettings,
        "clear:all" => MenuCommand::ClearHistory(History::ALL.to_vec()),
        "clear:logs" => MenuCommand::ClearHistory(vec![History::Logs]),
        "clear:traces" => MenuCommand::ClearHistory(vec![History::Traces]),
        "clear:trees" => MenuCommand::ClearHistory(vec![History::Trees]),
        "clear:recordings" => MenuCommand::ClearHistory(vec![History::Recordings]),
        "clear:machines" => MenuCommand::ClearHistory(vec![History::Machines]),
        "quit" => MenuCommand::Quit,
        other => {
            let branch = other.strip_prefix("start:")?;
            MenuCommand::StartAt((!branch.is_empty()).then(|| branch.to_string()))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machines_menu_names_each_machine_s_state_and_counts_the_tasks() {
        let entry = |id, label: &str, state: &str, task| MachineEntry {
            id,
            label: label.into(),
            state: state.into(),
            waiting: Vec::new(),
            task,
        };
        let root = entry(1, "/", "idle", false);
        let agent = entry(2, "research", "idle", false);
        let first = entry(3, "search-1", "results", true);
        let second = entry(4, "search-2", "reading", true);
        assert_eq!(machine_line(&first), "search-1 · results");
        assert_eq!(machine_line(&root), "/ · idle");
        assert_eq!(machines_title(&[]), "Machines");
        assert_eq!(machines_title(&[root.clone(), agent.clone()]), "Machines");
        assert_eq!(
            machines_title(&[root.clone(), agent.clone(), first.clone()]),
            "Machines · 1 task runs"
        );
        assert_eq!(
            machines_title(&[root, agent, first, second]),
            "Machines · 2 tasks run"
        );
    }

    #[test]
    fn menu_ids_map_to_commands() {
        assert_eq!(menu_command("toggle"), Some(MenuCommand::ToggleDictation));
        assert_eq!(menu_command("start:"), Some(MenuCommand::StartAt(None)));
        assert_eq!(
            menu_command("start:ask"),
            Some(MenuCommand::StartAt(Some("ask".into())))
        );
        assert_eq!(menu_command("start"), None);
        assert_eq!(
            menu_command("conversation"),
            Some(MenuCommand::ShowConversation)
        );
        assert_eq!(menu_command("feedback"), Some(MenuCommand::ToggleFeedback));
        assert_eq!(menu_command("cancel"), Some(MenuCommand::CancelTake));
        assert_eq!(menu_command("cancel:tasks"), Some(MenuCommand::CancelTasks));
        // A machine that runs is named by its id; a line that only says what a task waits
        // for, or the submenu itself, asks for nothing.
        assert_eq!(
            menu_command("machine:12"),
            Some(MenuCommand::ShowMachine(12))
        );
        assert_eq!(menu_command("end:12"), Some(MenuCommand::CancelOneTask(12)));
        assert_eq!(menu_command("machine:search"), None);
        assert_eq!(menu_command("waits:12"), None);
        assert_eq!(menu_command("task:12"), None);
        assert_eq!(menu_command("machines"), None);
        // The line that says automations are coming soon asks for nothing either.
        assert_eq!(menu_command("automations-soon"), None);
        assert_eq!(
            menu_command("clear:machines"),
            Some(MenuCommand::ClearHistory(vec![History::Machines]))
        );
        assert_eq!(menu_command("record"), Some(MenuCommand::ToggleRecording));
        assert_eq!(
            menu_command("record:discard"),
            Some(MenuCommand::DiscardRecording)
        );
        assert_eq!(
            menu_command("run:slack-post"),
            Some(MenuCommand::RunAutomation("slack-post".into()))
        );
        assert_eq!(
            menu_command("step:slack-post"),
            Some(MenuCommand::RunStepByStep("slack-post".into()))
        );
        assert_eq!(
            menu_command("again:slack-post"),
            Some(MenuCommand::RecordAgain("slack-post".into()))
        );
        assert_eq!(
            menu_command("approve:slack-post"),
            Some(MenuCommand::ApproveAutomation("slack-post".into()))
        );
        assert_eq!(
            menu_command("reset-settings"),
            Some(MenuCommand::ResetSettings)
        );
        assert_eq!(
            menu_command("clear:traces"),
            Some(MenuCommand::ClearHistory(vec![History::Traces]))
        );
        assert_eq!(
            menu_command("clear:all"),
            Some(MenuCommand::ClearHistory(History::ALL.to_vec()))
        );
    }
}
