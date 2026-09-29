//! Swallows the auto-repeats of a held hotkey. Windows delivers a held key's repeats to the
//! focused application (global hotkeys are registered without repeats), so holding F10 toggles
//! Notepad's menu bar about 30 times a second and the text being typed turns into menu shortcuts.
//! While a take runs from a held hotkey, a low-level keyboard hook drops the repeats of that key;
//! its release, and everything typed, pass through.

use std::sync::Mutex;

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
    use super::HELD;

    pub fn start() {
        let spawned = std::thread::Builder::new()
            .name("key-hook".into())
            .spawn(|| {
                let result = rdev::grab(|event| match event.event_type {
                    rdev::EventType::KeyPress(key) if is_held(key) => None,
                    _ => Some(event),
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
    fn key_name(key: rdev::Key) -> String {
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
