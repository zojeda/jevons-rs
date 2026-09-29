//! Windows: UI Automation for the focused element, SendInput (through enigo) and the clipboard
//! for delivery.

use super::window_snapshot;
use device_query::{DeviceQuery, DeviceState};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use jevons_desktop_core::context::{ContextSnapshot, Element, Privacy};
use jevons_desktop_core::platform::{
    ContextProvider, DeliveryOutcome, DeliveryRequest, PlatformError, SinkCapabilities, TextSink,
};
use jevons_desktop_core::profile::DeliveryMethod;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use uiautomation::patterns::{UITextPattern, UIValuePattern};
use uiautomation::types::{ControlType, TextPatternRangeEndpoint};
use uiautomation::{UIAutomation, UIElement};

/// Browsers whose address bar gives the page address.
const BROWSERS: &[&str] = &[
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "vivaldi.exe",
];

fn failed(e: impl std::fmt::Display) -> PlatformError {
    PlatformError::Failed(e.to_string())
}

type Reply<T> = std::sync::mpsc::Sender<Result<T, PlatformError>>;

enum Request {
    Snapshot(Privacy, Reply<ContextSnapshot>),
    SetValue(String, Reply<()>),
}

/// UI Automation calls, on one thread that owns its COM apartment: the threads that ask may
/// already have COM initialized another way (audio does), which UI Automation cannot share.
fn automation(request: Request) {
    static WORKER: OnceLock<Mutex<std::sync::mpsc::Sender<Request>>> = OnceLock::new();
    let worker = WORKER.get_or_init(|| {
        let (sender, requests) = std::sync::mpsc::channel::<Request>();
        std::thread::Builder::new()
            .name("ui-automation".into())
            .spawn(move || {
                let automation = UIAutomation::new().map_err(failed);
                for request in requests {
                    let automation = automation.as_ref().map_err(failed);
                    match request {
                        Request::Snapshot(privacy, reply) => {
                            let _ = reply.send(automation.and_then(|a| snapshot(a, &privacy)));
                        }
                        Request::SetValue(text, reply) => {
                            let _ = reply.send(automation.and_then(|a| set_value(a, &text)));
                        }
                    }
                }
            })
            .expect("the UI Automation thread starts");
        Mutex::new(sender)
    });
    let _ = worker.lock().expect("the UI Automation lock").send(request);
}

fn ask<T>(request: impl FnOnce(Reply<T>) -> Request) -> Result<T, PlatformError> {
    let (reply, answer) = std::sync::mpsc::channel();
    automation(request(reply));
    answer
        .recv()
        .map_err(|_| failed("the UI Automation thread stopped"))?
}

/// The focused element through UI Automation.
pub struct UiaContext;

impl ContextProvider for UiaContext {
    fn name(&self) -> &'static str {
        "UI Automation"
    }

    fn snapshot(&self, privacy: &Privacy) -> Result<ContextSnapshot, PlatformError> {
        ask(|reply| Request::Snapshot(privacy.clone(), reply))
    }
}

fn snapshot(
    automation: &UIAutomation,
    privacy: &Privacy,
) -> Result<ContextSnapshot, PlatformError> {
    let mut snapshot = window_snapshot()?;
    match automation.get_focused_element() {
        Ok(element) => {
            let max = privacy.max_context_chars as i32;
            let (focused, errors) = read_element(&element, max);
            snapshot.errors.extend(errors);
            if let Ok(class) = element.get_classname() {
                snapshot.extras.insert("class".into(), class);
            }
            snapshot.focused = Some(focused);
        }
        Err(e) => snapshot.errors.push(format!("No focused element: {e}")),
    }
    if BROWSERS.contains(&snapshot.app.process_name.to_lowercase().as_str()) {
        match address_bar(automation, snapshot.app.pid) {
            Some(url) => snapshot.url = Some(url),
            None => snapshot
                .errors
                .push("The browser address bar was not found".into()),
        }
    }
    if privacy.read_clipboard
        && let Ok(text) = arboard::Clipboard::new().and_then(|mut c| c.get_text())
    {
        snapshot.extras.insert("clipboard".into(), text);
    }
    Ok(snapshot.sanitized(privacy))
}

fn set_value(automation: &UIAutomation, text: &str) -> Result<(), PlatformError> {
    let element = automation.get_focused_element().map_err(failed)?;
    let value = element.get_pattern::<UIValuePattern>().map_err(failed)?;
    value.set_value(text).map_err(failed)
}

fn read_element(element: &UIElement, max: i32) -> (Element, Vec<String>) {
    let mut errors = Vec::new();
    let mut focused = Element {
        role: element
            .get_control_type()
            .map(|t| format!("{t:?}"))
            .unwrap_or_default(),
        name: element.get_name().unwrap_or_default(),
        automation_id: element.get_automation_id().ok().filter(|a| !a.is_empty()),
        is_password: element.is_password().unwrap_or(false),
        ..Element::default()
    };
    if focused.is_password {
        return (focused, errors);
    }
    if let Ok(value) = element.get_pattern::<UIValuePattern>() {
        focused.is_editable = !value.is_readonly().unwrap_or(true);
        focused.value_excerpt = value.get_value().ok().filter(|v| !v.is_empty());
    }
    match element.get_pattern::<UITextPattern>() {
        Ok(text) => {
            focused.is_editable |= matches!(
                element.get_control_type(),
                Ok(ControlType::Edit | ControlType::Document)
            );
            focused.selection = text
                .get_selection()
                .ok()
                .and_then(|ranges| ranges.first().and_then(|r| r.get_text(max).ok()))
                .filter(|s| !s.is_empty());
            match text.get_caret_range() {
                Ok((_, caret)) => {
                    focused.before_caret = text.get_document_range().ok().and_then(|before| {
                        before
                            .move_endpoint_by_range(
                                TextPatternRangeEndpoint::End,
                                &caret,
                                TextPatternRangeEndpoint::Start,
                            )
                            .ok()?;
                        before.get_text(-1).ok()
                    });
                    focused.after_caret = text.get_document_range().ok().and_then(|after| {
                        after
                            .move_endpoint_by_range(
                                TextPatternRangeEndpoint::Start,
                                &caret,
                                TextPatternRangeEndpoint::End,
                            )
                            .ok()?;
                        after.get_text(max).ok()
                    });
                }
                Err(e) => errors.push(format!("No caret: {e}")),
            }
            if focused.value_excerpt.is_none() {
                focused.value_excerpt = text
                    .get_document_range()
                    .and_then(|r| r.get_text(max))
                    .ok()
                    .filter(|v| !v.is_empty());
            }
        }
        Err(_) if focused.value_excerpt.is_none() => {
            errors.push("The focused element does not expose its text".into());
        }
        Err(_) => {}
    }
    (focused, errors)
}

/// The address bar text of the browser window owned by `pid`.
fn address_bar(automation: &UIAutomation, pid: Option<u32>) -> Option<String> {
    let root = automation.get_root_element().ok()?;
    let matcher = automation
        .create_matcher()
        .from(root)
        .control_type(ControlType::Edit)
        .depth(12)
        .timeout(0)
        .filter_fn(Box::new(move |e: &UIElement| {
            if pid.is_some_and(|pid| e.get_process_id().ok() != Some(pid)) {
                return Ok(false);
            }
            let name = e.get_name().unwrap_or_default().to_lowercase();
            Ok(name.contains("address") || name.contains("dirección") || name.contains("url"))
        }));
    let edit = matcher.find_first().ok()?;
    let value = edit
        .get_pattern::<UIValuePattern>()
        .ok()?
        .get_value()
        .ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    Some(if value.contains("://") {
        value.to_string()
    } else {
        format!("https://{value}")
    })
}

/// SendInput and the clipboard.
pub struct WindowsSink {
    keys: DeviceState,
}

impl WindowsSink {
    pub fn new() -> Self {
        Self {
            keys: DeviceState::new(),
        }
    }

    fn enigo() -> Result<Enigo, PlatformError> {
        Enigo::new(&Settings::default()).map_err(failed)
    }

    fn chord(enigo: &mut Enigo, letter: char) -> Result<(), PlatformError> {
        enigo.key(Key::Control, Direction::Press).map_err(failed)?;
        let result = enigo.key(Key::Unicode(letter), Direction::Click);
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(failed)?;
        result.map_err(failed)
    }

    /// Pastes `text`, restoring the previous clipboard text afterwards.
    /// Pastes `text` after deleting `erase` characters, restoring the previous clipboard text.
    fn paste(&mut self, text: &str, select_all: bool, erase: usize) -> Result<(), PlatformError> {
        let mut clipboard = arboard::Clipboard::new().map_err(failed)?;
        let previous = clipboard.get_text().ok();
        clipboard.set_text(text).map_err(failed)?;
        let mut enigo = Self::enigo()?;
        if select_all {
            Self::chord(&mut enigo, 'a')?;
        }
        // Live dictation replacing words it inserted that recognition revised.
        for _ in 0..erase {
            enigo
                .key(Key::Backspace, Direction::Click)
                .map_err(failed)?;
        }
        Self::chord(&mut enigo, 'v')?;
        // The target reads the clipboard asynchronously; give it time before restoring.
        std::thread::sleep(Duration::from_millis(300));
        if let Some(previous) = previous {
            let _ = clipboard.set_text(previous);
        }
        Ok(())
    }
}

impl TextSink for WindowsSink {
    fn name(&self) -> &'static str {
        "SendInput"
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            paste: true,
            type_text: true,
            set_value: true,
        }
    }

    fn foreground_window(&self) -> Option<u64> {
        super::active_window().map(|(_, handle)| handle)
    }

    fn keys_down(&self) -> bool {
        !self.keys.get_keys().is_empty()
    }

    fn deliver(&mut self, request: &DeliveryRequest) -> Result<DeliveryOutcome, PlatformError> {
        match request.method {
            DeliveryMethod::Paste => {
                self.paste(&request.text, request.select_all, request.erase)?
            }
            DeliveryMethod::Type => {
                let mut enigo = Self::enigo()?;
                if request.select_all {
                    Self::chord(&mut enigo, 'a')?;
                }
                // Live dictation replacing words it typed that recognition revised.
                for _ in 0..request.erase {
                    enigo
                        .key(Key::Backspace, Direction::Click)
                        .map_err(failed)?;
                }
                enigo.text(&request.text).map_err(failed)?;
            }
            DeliveryMethod::SetValue => {
                ask(|reply| Request::SetValue(request.text.clone(), reply))?
            }
            DeliveryMethod::Clipboard => {
                self.copy(&request.text)?;
                return Ok(DeliveryOutcome::OnClipboard {
                    reason: "the profile delivers to the clipboard".into(),
                });
            }
        }
        Ok(DeliveryOutcome::Delivered {
            method: request.method,
        })
    }

    fn copy(&mut self, text: &str) -> Result<(), PlatformError> {
        arboard::Clipboard::new()
            .and_then(|mut c| c.set_text(text))
            .map_err(failed)
    }
}
