//! The tray icon, its menu and the global hotkey, on their own thread with a tao event loop
//! (the eframe window keeps the main thread). The icon animates here: the waveform follows the
//! microphone level the agent reports, and the processing dots advance on a timer.

use crate::agent::Command;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use jevons_desktop_core::icons::{self, FRAME, TrayState};
use jevons_desktop_core::platform::{HotkeyEvent, MenuCommand, MenuModel, TrayBackend};
use std::time::Instant;
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::platform::run_return::EventLoopExtRunReturn;
use tokio::sync::mpsc::UnboundedSender;
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};

/// Messages to the tray thread.
#[derive(Debug)]
pub enum TrayMessage {
    State(TrayState),
    Menu(MenuModel),
    /// Registers this accelerator as the dictation hotkey, replacing the previous one.
    Hotkey(String),
    Quit,
}

/// Talks to the tray thread.
#[derive(Clone)]
pub struct Tray {
    proxy: EventLoopProxy<TrayMessage>,
}

impl Tray {
    pub fn hotkey(&self, accelerator: &str) {
        let _ = self
            .proxy
            .send_event(TrayMessage::Hotkey(accelerator.into()));
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
    let mut state = TrayState::Offline;
    let mut shown: Option<TrayState> = None;
    let mut menu = MenuModel::default();
    let mut hotkey: Option<HotKey> = None;
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
                    state = new;
                }
            }
            Event::UserEvent(TrayMessage::Menu(model)) => {
                menu = model;
                if let Some(tray) = &tray {
                    tray.set_menu(Some(Box::new(build_menu(&menu))));
                }
            }
            Event::UserEvent(TrayMessage::Hotkey(accelerator)) => {
                if let Some(manager) = &hotkeys {
                    if let Some(old) = hotkey.take() {
                        let _ = manager.unregister(old);
                    }
                    match accelerator.parse::<HotKey>() {
                        Ok(new) => match manager.register(new) {
                            Ok(()) => {
                                hotkey = Some(new);
                                let _ = commands.send(Command::HotkeyRegistered(Ok(new.id())));
                            }
                            Err(e) => {
                                let _ = commands.send(Command::HotkeyRegistered(Err(format!(
                                    "Cannot register {accelerator}: {e}"
                                ))));
                            }
                        },
                        Err(e) => {
                            let _ = commands.send(Command::HotkeyRegistered(Err(format!(
                                "Invalid hotkey {accelerator:?}: {e}"
                            ))));
                        }
                    }
                }
            }
            Event::UserEvent(TrayMessage::Quit) => {
                tray = None;
                *control_flow = ControlFlow::Exit;
                return;
            }
            _ => {}
        }
        if let Some(icon) = &tray
            && shown != Some(state)
        {
            let _ = icon.set_icon(Some(frames[icons::frame_index(state)].clone()));
            if shown.is_none_or(|s| std::mem::discriminant(&s) != std::mem::discriminant(&state)) {
                let _ = icon.set_tooltip(Some(state.tooltip()));
            }
            shown = Some(state);
        }
        if state.animates() {
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

fn build_menu(model: &MenuModel) -> Menu {
    let menu = Menu::new();
    let dictation = if model.dictating {
        "Stop dictation"
    } else {
        "Start dictation"
    };
    let profiles = Submenu::with_id("profiles", "Profile", true);
    let _ = profiles.append(&CheckMenuItem::with_id(
        "profile:",
        "Automatic",
        true,
        model.forced.is_none(),
        None,
    ));
    let _ = profiles.append(&PredefinedMenuItem::separator());
    for (id, name) in &model.profiles {
        let _ = profiles.append(&CheckMenuItem::with_id(
            format!("profile:{id}"),
            name,
            true,
            model.forced.as_ref() == Some(id),
            None,
        ));
    }
    let _ = menu.append_items(&[
        &MenuItem::with_id("toggle", dictation, true, None),
        &profiles,
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id("inspector", "Show context inspector", true, None),
        &CheckMenuItem::with_id(
            "pause",
            "Pause context capture",
            true,
            model.context_paused,
            None,
        ),
        &MenuItem::with_id("reload", "Reload profiles", true, None),
        &MenuItem::with_id("config", "Open settings folder", true, None),
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id("quit", "Quit", true, None),
    ]);
    menu
}

fn menu_command(id: &str) -> Option<MenuCommand> {
    Some(match id {
        "toggle" => MenuCommand::ToggleDictation,
        "inspector" => MenuCommand::ShowInspector,
        "pause" => MenuCommand::ToggleContextPause,
        "reload" => MenuCommand::ReloadProfiles,
        "config" => MenuCommand::OpenConfigFolder,
        "quit" => MenuCommand::Quit,
        other => {
            let profile = other.strip_prefix("profile:")?;
            MenuCommand::ForceProfile((!profile.is_empty()).then(|| profile.to_string()))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_ids_map_to_commands() {
        assert_eq!(menu_command("toggle"), Some(MenuCommand::ToggleDictation));
        assert_eq!(
            menu_command("profile:"),
            Some(MenuCommand::ForceProfile(None))
        );
        assert_eq!(
            menu_command("profile:slack"),
            Some(MenuCommand::ForceProfile(Some("slack".into())))
        );
        assert_eq!(menu_command("profiles"), None);
    }
}
