//! Swallows the auto-repeats of a held hotkey. Windows delivers a held key's repeats to the
//! focused application (global hotkeys are registered without repeats), so holding F10 toggles
//! Notepad's menu bar about 30 times a second and the text being typed turns into menu shortcuts.
//! While a take runs from a held hotkey, a low-level keyboard hook drops the repeats of that key;
//! its release, and everything typed, pass through.
//!
//! The same hook reports clicks and keys while a demonstration is recorded ([`record`]),
//! except the input the app sends itself ([`own_input`]).

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Input the hook saw while a demonstration is recorded.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(not(windows), allow(dead_code))]
pub enum Raw {
    /// The left mouse button went down at this screen point.
    Click { x: i32, y: i32 },
    /// A key went down.
    Key {
        /// The key's name as accelerators write it: `a`, `enter`, `f5`.
        key: String,
        /// The text it types (shift and the layout applied), when it types any.
        text: Option<String>,
        ctrl: bool,
        alt: bool,
        shift: bool,
        meta: bool,
    },
}

/// Where the hook sends input while recording.
static RECORDING: Mutex<Option<std::sync::mpsc::Sender<Raw>>> = Mutex::new(None);
/// Until when input is the app's own (it just typed or clicked), and not the user's.
static OWN_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);

/// Starts (with a sender) or stops (with `None`) reporting input.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn record(sender: Option<std::sync::mpsc::Sender<Raw>>) {
    *RECORDING.lock().expect("the recording lock") = sender;
}

/// Marks the input of the next moments as the app's own: call it before and after the app
/// types, pastes or clicks. The hook sees injected input a little after it is sent.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn own_input() {
    *OWN_UNTIL.lock().expect("the own input lock") =
        Some(Instant::now() + Duration::from_millis(400));
}

#[cfg_attr(not(windows), allow(dead_code))]
fn is_own() -> bool {
    OWN_UNTIL
        .lock()
        .expect("the own input lock")
        .is_some_and(|until| Instant::now() < until)
}

#[cfg_attr(not(windows), allow(dead_code))]
fn report(raw: Raw) {
    if is_own() {
        return;
    }
    if let Some(sender) = RECORDING.lock().expect("the recording lock").as_ref() {
        let _ = sender.send(raw);
    }
}

/// The key of the hotkey held for the running take, as the last part of its accelerator.
static HELD: Mutex<Option<String>> = Mutex::new(None);

/// Drops the repeats of `accelerator`'s key until [`release`].
pub fn hold(accelerator: &str) {
    let key = accelerator.rsplit('+').next().unwrap_or(accelerator).trim();
    *HELD.lock().expect("the held key lock") = Some(key.to_ascii_lowercase());
}

pub fn release() {
    *HELD.lock().expect("the held key lock") = None;
}

/// Starts the keyboard hook on its own thread (Windows only).
pub fn start() {
    #[cfg(windows)]
    windows::start();
}

#[cfg(windows)]
mod windows {
    use super::{HELD, Raw, report};
    use std::sync::Mutex;

    /// The pointer's last position and the modifiers held, for what the hook reports.
    #[derive(Default)]
    struct Input {
        x: f64,
        y: f64,
        ctrl: bool,
        alt: bool,
        shift: bool,
        meta: bool,
    }

    static INPUT: Mutex<Input> = Mutex::new(Input {
        x: 0.0,
        y: 0.0,
        ctrl: false,
        alt: false,
        shift: false,
        meta: false,
    });

    /// Tracks the pointer and modifiers, and reports clicks and keys while recording. It must
    /// be quick: the hook holds up all input while it runs.
    fn observe(event: &rdev::Event) {
        use rdev::{Button, EventType, Key};
        let mut input = INPUT.lock().expect("the input lock");
        match event.event_type {
            EventType::MouseMove { x, y } => {
                input.x = x;
                input.y = y;
            }
            EventType::ButtonPress(Button::Left) => report(Raw::Click {
                x: input.x.round() as i32,
                y: input.y.round() as i32,
            }),
            EventType::KeyPress(key) | EventType::KeyRelease(key) => {
                let down = matches!(event.event_type, EventType::KeyPress(_));
                match key {
                    Key::ControlLeft | Key::ControlRight => input.ctrl = down,
                    Key::Alt | Key::AltGr => input.alt = down,
                    Key::ShiftLeft | Key::ShiftRight => input.shift = down,
                    Key::MetaLeft | Key::MetaRight => input.meta = down,
                    _ if down => report(Raw::Key {
                        key: key_name(key),
                        text: event
                            .name
                            .clone()
                            .filter(|t| !t.chars().any(char::is_control)),
                        ctrl: input.ctrl,
                        alt: input.alt,
                        shift: input.shift,
                        meta: input.meta,
                    }),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    pub fn start() {
        let spawned = std::thread::Builder::new()
            .name("key-hook".into())
            .spawn(|| {
                let result = rdev::grab(|event| {
                    observe(&event);
                    match event.event_type {
                        rdev::EventType::KeyPress(key) if is_held(key) => None,
                        _ => Some(event),
                    }
                });
                if let Err(e) = result {
                    tracing::warn!(error = ?e, "The keyboard hook stopped: held hotkeys repeat into the focused application");
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "Cannot start the keyboard hook");
        }
    }

    fn is_held(key: rdev::Key) -> bool {
        let held = HELD.lock().expect("the held key lock");
        held.as_deref().is_some_and(|name| name == key_name(key))
    }

    /// The accelerator name (as global-hotkey writes it, lowercased) of an rdev key.
    pub(super) fn key_name(key: rdev::Key) -> String {
        use rdev::Key::*;
        let name = match key {
            Space => "space",
            Return => "enter",
            Tab => "tab",
            Escape => "escape",
            Backspace => "backspace",
            Insert => "insert",
            Delete => "delete",
            Home => "home",
            End => "end",
            PageUp => "pageup",
            PageDown => "pagedown",
            UpArrow => "arrowup",
            DownArrow => "arrowdown",
            LeftArrow => "arrowleft",
            RightArrow => "arrowright",
            Pause => "pause",
            ScrollLock => "scrolllock",
            PrintScreen => "printscreen",
            other => {
                // F1..F12, KeyA..KeyZ and Num0..Num9 map to "f1", "a", "1".
                let debug = format!("{other:?}");
                let name = debug
                    .strip_prefix("Key")
                    .or_else(|| debug.strip_prefix("Num"))
                    .unwrap_or(&debug);
                return name.to_ascii_lowercase();
            }
        };
        name.to_string()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn hotkey_keys_match_their_rdev_keys() {
            assert_eq!(key_name(rdev::Key::F10), "f10");
            assert_eq!(key_name(rdev::Key::KeyL), "l");
            assert_eq!(key_name(rdev::Key::Num1), "1");
            assert_eq!(key_name(rdev::Key::Space), "space");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_held_key_is_the_accelerator_s_last_part() {
        hold("Ctrl+Alt+Space");
        assert_eq!(HELD.lock().unwrap().as_deref(), Some("space"));
        hold("F10");
        assert_eq!(HELD.lock().unwrap().as_deref(), Some("f10"));
        release();
        assert_eq!(*HELD.lock().unwrap(), None);
    }
}
