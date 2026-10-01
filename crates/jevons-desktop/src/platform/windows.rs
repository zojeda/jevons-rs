//! Windows: UI Automation for the focused element, the interfaces investigations and XPath read
//! and the actions automations take; SendInput (through enigo) and the clipboard for delivery,
//! keys and clicks.

use super::window_snapshot;
use device_query::{DeviceQuery, DeviceState};
use enigo::{Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use jevons_desktop_core::context::{ContextSnapshot, Element, Privacy};
use jevons_desktop_core::platform::{
    Acted, Chord, Condition, ContextInspector, ContextProvider, DeliveryMethod, DeliveryOutcome,
    DeliveryRequest, Modifier, Observed, PlatformError, Property, Reach, Recorder, RecordingHandle,
    SinkCapabilities, TextSink, UiAction, UiActor, UiElement, WindowEntry,
};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use uiautomation::core::{UICacheRequest, UICondition};
use uiautomation::patterns::{
    UIExpandCollapsePattern, UIInvokePattern, UILegacyIAccessiblePattern, UIScrollItemPattern,
    UISelectionItemPattern, UITextPattern, UITogglePattern, UIValuePattern, UIWindowPattern,
};
use uiautomation::types::{
    ControlType, PropertyConditionFlags, TextPatternRangeEndpoint, TreeScope, UIProperty,
    WindowVisualState,
};
use uiautomation::variants::Variant;
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
    Windows(Reply<Vec<WindowEntry>>),
    Children(String, Reply<Vec<UiElement>>),
    Subtree(String, usize, usize, Reply<Vec<(usize, UiElement)>>),
    Parent(String, Reply<Option<UiElement>>),
    Find(String, Reach, Vec<Condition>, usize, Reply<Vec<UiElement>>),
    Act(String, UiAction, Reply<Acted>),
    Activate(String, Reply<()>),
    ElementAt(i32, i32, Reply<Option<(UiElement, WindowEntry)>>),
    Focused(Reply<Option<(UiElement, WindowEntry)>>),
}

/// The most elements the worker keeps references to; past it they are dropped and callers
/// that still hold ids read the windows again.
const MAX_ELEMENTS: usize = 50_000;

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
                // Elements handed out by id, so their children and parents can be read later.
                let mut elements: HashMap<String, UIElement> = HashMap::new();
                for request in requests {
                    let automation = automation.as_ref().map_err(failed);
                    if elements.len() > MAX_ELEMENTS {
                        elements.clear();
                    }
                    match request {
                        Request::Snapshot(privacy, reply) => {
                            let _ = reply.send(
                                automation.and_then(|a| snapshot(a, &privacy, &mut elements)),
                            );
                        }
                        Request::SetValue(text, reply) => {
                            let _ = reply.send(automation.and_then(|a| set_value(a, &text)));
                        }
                        Request::Windows(reply) => {
                            let _ = reply.send(automation.and_then(|a| windows(a, &mut elements)));
                        }
                        Request::Parent(id, reply) => {
                            let _ =
                                reply.send(automation.and_then(|a| parent(a, &mut elements, &id)));
                        }
                        Request::Find(id, reach, conditions, limit, reply) => {
                            let _ = reply.send(automation.and_then(|a| {
                                find(a, &mut elements, &id, reach, &conditions, limit)
                            }));
                        }
                        Request::Act(id, action, reply) => {
                            let _ = reply
                                .send(element_of(&elements, &id).and_then(|e| act(&e, &action)));
                        }
                        Request::ElementAt(x, y, reply) => {
                            let _ = reply
                                .send(automation.and_then(|a| element_at(a, &mut elements, x, y)));
                        }
                        Request::Focused(reply) => {
                            let _ = reply.send(automation.and_then(|a| focused(a, &mut elements)));
                        }
                        Request::Activate(id, reply) => {
                            let _ =
                                reply.send(element_of(&elements, &id).and_then(|e| activate(&e)));
                        }
                        Request::Children(id, reply) => {
                            let _ = reply
                                .send(automation.and_then(|a| children(a, &mut elements, &id)));
                        }
                        Request::Subtree(id, depth, limit, reply) => {
                            let _ =
                                reply
                                    .send(automation.and_then(|a| {
                                        subtree(a, &mut elements, &id, depth, limit)
                                    }));
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

/// Interfaces read on demand, for the context investigator.
pub struct UiaInspector;

impl ContextInspector for UiaInspector {
    fn name(&self) -> &'static str {
        "UI Automation"
    }

    fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
        ask(Request::Windows)
    }

    fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError> {
        ask(|reply| Request::Children(id.to_string(), reply))
    }

    fn subtree(
        &self,
        id: &str,
        depth: usize,
        limit: usize,
    ) -> Result<Vec<(usize, UiElement)>, PlatformError> {
        ask(|reply| Request::Subtree(id.to_string(), depth, limit, reply))
    }

    fn parent(&self, id: &str) -> Result<Option<UiElement>, PlatformError> {
        ask(|reply| Request::Parent(id.to_string(), reply))
    }

    fn focused(&self) -> Result<Option<(UiElement, WindowEntry)>, PlatformError> {
        ask(Request::Focused)
    }

    fn find(
        &self,
        id: &str,
        reach: Reach,
        conditions: &[Condition],
        limit: usize,
    ) -> Result<Vec<UiElement>, PlatformError> {
        ask(|reply| Request::Find(id.to_string(), reach, conditions.to_vec(), limit, reply))
    }
}

/// The top-level window an element is in, as the window list describes it.
fn window_of(
    automation: &UIAutomation,
    element: &UIElement,
    elements: &mut HashMap<String, UIElement>,
) -> Result<Option<WindowEntry>, PlatformError> {
    let walker = automation.get_control_view_walker().map_err(failed)?;
    let root = automation.get_root_element().map_err(failed)?;
    let mut at = element.clone();
    loop {
        let Ok(parent) = walker.get_parent(&at) else {
            return Ok(None);
        };
        if automation.compare_elements(&parent, &root).unwrap_or(false) {
            break;
        }
        at = parent;
    }
    let Some(id) = element_id(&at) else {
        return Ok(None);
    };
    elements.insert(id.clone(), at.clone());
    let pid = at.get_process_id().unwrap_or_default();
    let mut system = sysinfo::System::new();
    system.refresh_processes(
        sysinfo::ProcessesToUpdate::Some(&[sysinfo::Pid::from_u32(pid)]),
        true,
    );
    let app = system
        .process(sysinfo::Pid::from_u32(pid))
        .map(|p| p.name().to_string_lossy().into_owned())
        .unwrap_or_default();
    let front = super::active_window().map(|(w, _)| w.process_id);
    Ok(Some(WindowEntry {
        id,
        app,
        title: at.get_name().unwrap_or_default(),
        front: front == Some(u64::from(pid)),
    }))
}

/// The element under a screen point, and its window.
fn element_at(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
    x: i32,
    y: i32,
) -> Result<Option<(UiElement, WindowEntry)>, PlatformError> {
    let request = cache_request(automation, TreeScope::Element)?;
    let Ok(raw) =
        automation.element_from_point_build_cache(uiautomation::types::Point::new(x, y), &request)
    else {
        return Ok(None);
    };
    // The point may land on an element the control view leaves out.
    let walker = automation.get_control_view_walker().map_err(failed)?;
    let element = walker.normalize_build_cache(&raw, &request).unwrap_or(raw);
    let Some(described) = describe(&element, elements) else {
        return Ok(None);
    };
    Ok(window_of(automation, &element, elements)?.map(|w| (described, w)))
}

/// The element with the keyboard focus, and its window.
fn focused(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
) -> Result<Option<(UiElement, WindowEntry)>, PlatformError> {
    let request = cache_request(automation, TreeScope::Element)?;
    let Ok(element) = automation.get_focused_element_build_cache(&request) else {
        return Ok(None);
    };
    let Some(described) = describe(&element, elements) else {
        return Ok(None);
    };
    Ok(window_of(automation, &element, elements)?.map(|w| (described, w)))
}

/// Records the user's clicks and keys from the keyboard hook, with the element under each
/// click.
pub struct UiaRecorder;

struct Recording {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl RecordingHandle for Recording {
    fn stop(self: Box<Self>) {
        crate::hold::record(None);
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Recorder for UiaRecorder {
    fn name(&self) -> &'static str {
        "UI Automation and a low-level hook"
    }

    fn start(
        &self,
        events: tokio::sync::mpsc::UnboundedSender<Observed>,
    ) -> Result<Box<dyn RecordingHandle>, PlatformError> {
        let (sender, raw) = std::sync::mpsc::channel::<crate::hold::Raw>();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopped = stop.clone();
        std::thread::Builder::new()
            .name("recorder".into())
            .spawn(move || {
                use std::sync::atomic::Ordering;
                while !stopped.load(Ordering::Relaxed) {
                    let Ok(input) = raw.recv_timeout(Duration::from_millis(200)) else {
                        continue;
                    };
                    let observed = match input {
                        crate::hold::Raw::Click { x, y } => {
                            match ask(|reply| Request::ElementAt(x, y, reply)) {
                                Ok(Some((element, window))) => Some(Observed::Click {
                                    element: Box::new(element),
                                    window,
                                }),
                                _ => None,
                            }
                        }
                        crate::hold::Raw::Key {
                            key,
                            text,
                            ctrl,
                            alt,
                            shift,
                            meta,
                        } => observed_key(&key, text.as_deref(), ctrl, alt, shift, meta),
                    };
                    if let Some(observed) = observed
                        && events.send(observed).is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(failed)?;
        crate::hold::record(Some(sender));
        Ok(Box::new(Recording { stop }))
    }
}

/// A key as the recording sees it: a chord with ctrl, alt or meta, a named key, or typed text.
fn observed_key(
    key: &str,
    text: Option<&str>,
    ctrl: bool,
    alt: bool,
    shift: bool,
    meta: bool,
) -> Option<Observed> {
    let named = !matches!(key.chars().count(), 1) && key != "space";
    if !(ctrl || alt || meta) && !named {
        let c = text?.chars().next()?;
        return Some(Observed::Char(c));
    }
    if !(ctrl || alt || meta) && key == "space" {
        return Some(Observed::Char(' '));
    }
    let mut accelerator = Vec::new();
    for (held, name) in [
        (ctrl, "ctrl"),
        (alt, "alt"),
        (shift, "shift"),
        (meta, "meta"),
    ] {
        if held {
            accelerator.push(name);
        }
    }
    let key = match key {
        "arrowup" => "up",
        "arrowdown" => "down",
        "arrowleft" => "left",
        "arrowright" => "right",
        other => other,
    };
    accelerator.push(key);
    Chord::parse(&accelerator.join("+"))
        .ok()
        .map(Observed::Chord)
}

/// Actions on elements found through [`UiaInspector`], and keys and text for the window in front.
pub struct UiaActor;

impl UiActor for UiaActor {
    fn name(&self) -> &'static str {
        "UI Automation"
    }

    fn act(&self, id: &str, action: &UiAction) -> Result<Acted, PlatformError> {
        crate::hold::own_input();
        let _mark = OwnInput;
        ask(|reply| Request::Act(id.to_string(), action.clone(), reply))
    }

    fn press(&self, chord: &Chord) -> Result<(), PlatformError> {
        crate::hold::own_input();
        let _mark = OwnInput;
        let mut enigo = WindowsSink::enigo()?;
        let modifiers: Vec<Key> = chord
            .modifiers
            .iter()
            .map(|m| match m {
                Modifier::Ctrl => Key::Control,
                Modifier::Alt => Key::Alt,
                Modifier::Shift => Key::Shift,
                Modifier::Meta => Key::Meta,
            })
            .collect();
        for modifier in &modifiers {
            enigo.key(*modifier, Direction::Press).map_err(failed)?;
        }
        let pressed = enigo.key(key_of(&chord.key), Direction::Click);
        // Modifiers are released even when the key failed, so none stays held down.
        for modifier in modifiers.iter().rev() {
            let _ = enigo.key(*modifier, Direction::Release);
        }
        pressed.map_err(failed)
    }

    fn type_text(&self, text: &str) -> Result<(), PlatformError> {
        crate::hold::own_input();
        let _mark = OwnInput;
        WindowsSink::enigo()?.text(text).map_err(failed)
    }

    fn activate(&self, window: &str) -> Result<(), PlatformError> {
        ask(|reply| Request::Activate(window.to_string(), reply))
    }

    fn front_app(&self) -> Option<String> {
        let (window, _) = super::active_window()?;
        window
            .process_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .or(Some(window.app_name))
    }
}

fn key_of(key: &jevons_desktop_core::platform::Key) -> Key {
    use jevons_desktop_core::platform::Key as K;
    const FUNCTION: [Key; 24] = [
        Key::F1,
        Key::F2,
        Key::F3,
        Key::F4,
        Key::F5,
        Key::F6,
        Key::F7,
        Key::F8,
        Key::F9,
        Key::F10,
        Key::F11,
        Key::F12,
        Key::F13,
        Key::F14,
        Key::F15,
        Key::F16,
        Key::F17,
        Key::F18,
        Key::F19,
        Key::F20,
        Key::F21,
        Key::F22,
        Key::F23,
        Key::F24,
    ];
    match key {
        K::Char(c) => Key::Unicode(*c),
        K::Enter => Key::Return,
        K::Tab => Key::Tab,
        K::Escape => Key::Escape,
        K::Backspace => Key::Backspace,
        K::Delete => Key::Delete,
        K::Insert => Key::Insert,
        K::Space => Key::Space,
        K::Up => Key::UpArrow,
        K::Down => Key::DownArrow,
        K::Left => Key::LeftArrow,
        K::Right => Key::RightArrow,
        K::Home => Key::Home,
        K::End => Key::End,
        K::PageUp => Key::PageUp,
        K::PageDown => Key::PageDown,
        K::Function(n) => FUNCTION[usize::from(n.clamp(&1, &24) - 1)],
    }
}

/// Marks input as the app's own again when it goes out of scope, after the input it covers.
struct OwnInput;

impl Drop for OwnInput {
    fn drop(&mut self) {
        crate::hold::own_input();
    }
}

/// Whether the window in front belongs to `element`'s process, waiting up to half a second for
/// a window just brought forward.
fn in_front(element: &UIElement) -> bool {
    let Ok(pid) = element.get_process_id() else {
        return false;
    };
    for _ in 0..10 {
        if super::active_window().is_some_and(|(w, _)| w.process_id == u64::from(pid)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// A mouse click at the element's clickable point, or the middle of its bounds.
fn click(element: &UIElement) -> Result<Acted, PlatformError> {
    let (x, y) = element
        .get_clickable_point()
        .ok()
        .flatten()
        .map(|p| (p.get_x(), p.get_y()))
        .or_else(|| {
            element.get_bounding_rectangle().ok().and_then(|r| {
                (r.get_right() > r.get_left() && r.get_bottom() > r.get_top()).then(|| {
                    (
                        (r.get_left() + r.get_right()) / 2,
                        (r.get_top() + r.get_bottom()) / 2,
                    )
                })
            })
        })
        .ok_or_else(|| failed("it has no point on screen to click (is it scrolled away?)"))?;
    let mut enigo = WindowsSink::enigo()?;
    enigo.move_mouse(x, y, Coordinate::Abs).map_err(failed)?;
    enigo
        .button(Button::Left, Direction::Click)
        .map_err(failed)?;
    Ok(Acted {
        how: format!("a click at {x}, {y}"),
    })
}

fn done(how: &str) -> Result<Acted, PlatformError> {
    Ok(Acted { how: how.into() })
}

/// One action on an element, through its control pattern when it has one.
fn act(element: &UIElement, action: &UiAction) -> Result<Acted, PlatformError> {
    let missing = |pattern: &str| failed(format!("it has no {pattern} pattern"));
    match action {
        // Activation: invoke, else select (list and tree items), else the default action,
        // else a click.
        UiAction::Invoke => {
            if let Ok(p) = element.get_pattern::<UIInvokePattern>() {
                p.invoke().map_err(failed)?;
                return done("the invoke pattern");
            }
            if let Ok(p) = element.get_pattern::<UISelectionItemPattern>() {
                p.select().map_err(failed)?;
                return done("the selection item pattern");
            }
            if let Ok(p) = element.get_pattern::<UILegacyIAccessiblePattern>()
                && p.do_default_action().is_ok()
            {
                return done("its default action");
            }
            click(element)
        }
        UiAction::Click => click(element),
        UiAction::Focus => {
            element.set_focus().map_err(failed)?;
            done("the focus")
        }
        UiAction::SetValue(text) => {
            let p = element
                .get_pattern::<UIValuePattern>()
                .map_err(|_| missing("value"))?;
            p.set_value(text).map_err(failed)?;
            done("the value pattern")
        }
        UiAction::TypeText(text) => {
            element.set_focus().map_err(failed)?;
            // Keys go to the window in front: never type unless it is the element's.
            if !in_front(element) {
                return Err(failed(
                    "its window did not come to the front, so nothing was typed",
                ));
            }
            WindowsSink::enigo()?.text(text).map_err(failed)?;
            done("the focus and typing")
        }
        UiAction::Toggle => {
            let p = element
                .get_pattern::<UITogglePattern>()
                .map_err(|_| missing("toggle"))?;
            p.toggle().map_err(failed)?;
            done("the toggle pattern")
        }
        UiAction::Select => {
            let p = element
                .get_pattern::<UISelectionItemPattern>()
                .map_err(|_| missing("selection item"))?;
            p.select().map_err(failed)?;
            done("the selection item pattern")
        }
        UiAction::Expand | UiAction::Collapse => {
            let p = element
                .get_pattern::<UIExpandCollapsePattern>()
                .map_err(|_| missing("expand/collapse"))?;
            if *action == UiAction::Expand {
                p.expand().map_err(failed)?;
            } else {
                p.collapse().map_err(failed)?;
            }
            done("the expand/collapse pattern")
        }
        UiAction::ScrollIntoView => {
            let p = element
                .get_pattern::<UIScrollItemPattern>()
                .map_err(|_| missing("scroll item"))?;
            p.scroll_into_view().map_err(failed)?;
            done("the scroll item pattern")
        }
    }
}

/// Brings a top-level window forward: restored if minimized, then focused.
fn activate(window: &UIElement) -> Result<(), PlatformError> {
    if let Ok(p) = window.get_pattern::<UIWindowPattern>()
        && matches!(
            p.get_window_visual_state(),
            Ok(WindowVisualState::Minimized)
        )
    {
        p.set_window_visual_state(WindowVisualState::Normal)
            .map_err(failed)?;
    }
    window.set_focus().map_err(failed)?;
    if in_front(window) {
        Ok(())
    } else {
        Err(failed(
            "Windows did not let the window come to the front (another application holds the \
             foreground)",
        ))
    }
}

/// The most characters of an element's value an investigation reads.
const MAX_VALUE: i32 = 4000;

fn element_id(element: &UIElement) -> Option<String> {
    let id = element.get_runtime_id().ok()?;
    Some(id.iter().map(i32::to_string).collect::<Vec<_>>().join("."))
}

/// The properties every element read for investigations and XPath carries, fetched in the
/// same cross-process call as the element itself.
const CACHED: &[UIProperty] = &[
    // Element ids: without it, each id would be one more cross-process call.
    UIProperty::RuntimeId,
    UIProperty::Name,
    UIProperty::ControlType,
    UIProperty::ClassName,
    UIProperty::AutomationId,
    UIProperty::IsPassword,
    UIProperty::IsEnabled,
    UIProperty::IsOffscreen,
    UIProperty::ValueValue,
    UIProperty::SelectionItemIsSelected,
    UIProperty::ToggleToggleState,
    UIProperty::ExpandCollapseExpandCollapseState,
];

/// A cache request for [`CACHED`] over the control view, reaching `scope`.
fn cache_request(
    automation: &UIAutomation,
    scope: TreeScope,
) -> Result<UICacheRequest, PlatformError> {
    let request = automation.create_cache_request().map_err(failed)?;
    for property in CACHED {
        request.add_property(*property).map_err(failed)?;
    }
    request
        .set_tree_filter(automation.get_control_view_condition().map_err(failed)?)
        .map_err(failed)?;
    request.set_tree_scope(scope).map_err(failed)?;
    Ok(request)
}

fn cached_bool(element: &UIElement, property: UIProperty) -> Option<bool> {
    let value = element.get_cached_property_value(property).ok()?;
    TryInto::<bool>::try_into(&value).ok()
}

fn cached_int(element: &UIElement, property: UIProperty) -> Option<i32> {
    let value = element.get_cached_property_value(property).ok()?;
    TryInto::<i32>::try_into(&value).ok()
}

/// An element from its cached properties; remembered so its children can be read later.
fn describe(element: &UIElement, elements: &mut HashMap<String, UIElement>) -> Option<UiElement> {
    let id = element_id(element)?;
    let password = cached_bool(element, UIProperty::IsPassword).unwrap_or(false);
    let control = element.get_cached_control_type().ok();
    let value = if password {
        None
    } else {
        element
            .get_cached_property_value(UIProperty::ValueValue)
            .ok()
            .and_then(|v| TryInto::<String>::try_into(&v).ok())
            .or_else(|| {
                // Documents give their text through the text pattern, which is not cached.
                matches!(control, Some(ControlType::Document))
                    .then(|| element.get_pattern::<UITextPattern>().ok())
                    .flatten()
                    .and_then(|t| {
                        t.get_document_range()
                            .and_then(|r| r.get_text(MAX_VALUE))
                            .ok()
                    })
            })
            .filter(|v| !v.is_empty())
            .map(|v| v.chars().take(MAX_VALUE as usize).collect())
    };
    let described = UiElement {
        id: id.clone(),
        role: control.map(|t| format!("{t:?}")).unwrap_or_default(),
        name: if password {
            String::new()
        } else {
            element.get_cached_name().unwrap_or_default()
        },
        value,
        class: element
            .get_cached_classname()
            .ok()
            .filter(|c| !c.is_empty()),
        automation_id: element
            .get_cached_automation_id()
            .ok()
            .filter(|a| !a.is_empty()),
        password,
        child_count: None,
        enabled: cached_bool(element, UIProperty::IsEnabled),
        offscreen: cached_bool(element, UIProperty::IsOffscreen),
        selected: cached_bool(element, UIProperty::SelectionItemIsSelected),
        // Off, on, indeterminate.
        toggled: cached_int(element, UIProperty::ToggleToggleState).map(|t| t == 1),
        // Collapsed, expanded, partially expanded; a leaf node (3) has no state.
        expanded: cached_int(element, UIProperty::ExpandCollapseExpandCollapseState)
            .filter(|s| *s != 3)
            .map(|s| s != 0),
    };
    elements.insert(id, element.clone());
    Some(described)
}

fn element_of(elements: &HashMap<String, UIElement>, id: &str) -> Result<UIElement, PlatformError> {
    elements
        .get(id)
        .cloned()
        .ok_or_else(|| failed(format!("no element {id}: list the windows again")))
}

/// UI Automation's control type ids by name (`ListItem`), as `UiElement::role` spells them.
fn control_type_id(role: &str) -> Option<i32> {
    static IDS: OnceLock<HashMap<String, i32>> = OnceLock::new();
    IDS.get_or_init(|| {
        // UIA_ButtonControlTypeId (50000) to UIA_AppBarControlTypeId (50040).
        (50_000..=50_040)
            .filter_map(|id| {
                ControlType::try_from(id)
                    .ok()
                    .map(|t| (format!("{t:?}"), id))
            })
            .collect()
    })
    .get(role)
    .copied()
}

/// The native condition for `conditions`, within the control view.
fn condition(
    automation: &UIAutomation,
    conditions: &[Condition],
) -> Result<UICondition, PlatformError> {
    let mut all = automation.get_control_view_condition().map_err(failed)?;
    for wanted in conditions {
        let (property, value) = match wanted.property {
            Property::Role => match control_type_id(&wanted.value) {
                Some(id) => (UIProperty::ControlType, Variant::from(id)),
                // A role UI Automation does not have: nothing matches it.
                None => return automation.create_false_condition().map_err(failed),
            },
            Property::Name => (UIProperty::Name, Variant::from(wanted.value.as_str())),
            Property::Class => (UIProperty::ClassName, Variant::from(wanted.value.as_str())),
            Property::AutomationId => (
                UIProperty::AutomationId,
                Variant::from(wanted.value.as_str()),
            ),
        };
        let flags = (wanted.substring && wanted.property != Property::Role)
            .then_some(PropertyConditionFlags::MatchSubstring);
        let one = automation
            .create_property_condition(property, value, flags)
            .map_err(failed)?;
        all = automation.create_and_condition(all, one).map_err(failed)?;
    }
    Ok(all)
}

/// The children of an element, in one cached call.
fn children(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
    id: &str,
) -> Result<Vec<UiElement>, PlatformError> {
    find(automation, elements, id, Reach::Children, &[], usize::MAX)
}

/// A native search below an element, with the properties cached in the same call.
fn find(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
    id: &str,
    reach: Reach,
    conditions: &[Condition],
    limit: usize,
) -> Result<Vec<UiElement>, PlatformError> {
    let parent = element_of(elements, id)?;
    let scope = match reach {
        Reach::Children => TreeScope::Children,
        Reach::Descendants => TreeScope::Descendants,
    };
    let request = cache_request(automation, TreeScope::Element)?;
    // UI Automation reports an empty result as an error on some elements.
    let found = parent
        .find_all_build_cache(scope, &condition(automation, conditions)?, &request)
        .unwrap_or_default();
    Ok(found
        .iter()
        .take(limit)
        .filter_map(|e| describe(e, elements))
        .collect())
}

/// The parent of an element, `None` below the desktop (a top-level window).
fn parent(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
    id: &str,
) -> Result<Option<UiElement>, PlatformError> {
    let element = element_of(elements, id)?;
    let walker = automation.get_control_view_walker().map_err(failed)?;
    let request = cache_request(automation, TreeScope::Element)?;
    let Ok(parent) = walker.get_parent_build_cache(&element, &request) else {
        return Ok(None);
    };
    let root = automation.get_root_element().map_err(failed)?;
    if automation.compare_elements(&parent, &root).unwrap_or(false) {
        return Ok(None);
    }
    Ok(describe(&parent, elements))
}

/// A subtree in one cross-process call, parents before children.
fn subtree(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
    id: &str,
    depth: usize,
    limit: usize,
) -> Result<Vec<(usize, UiElement)>, PlatformError> {
    let root = element_of(elements, id)?;
    let request = cache_request(automation, TreeScope::Subtree)?;
    let cached = root.build_updated_cache(&request).map_err(failed)?;
    let mut out = Vec::new();
    // A leaf reports its cached children as an error: it has none.
    let mut stack: Vec<(usize, UIElement)> = cached
        .get_cached_children()
        .unwrap_or_default()
        .into_iter()
        .rev()
        .map(|e| (1, e))
        .collect();
    while let Some((level, element)) = stack.pop() {
        if out.len() >= limit {
            break;
        }
        let below = if level < depth {
            element.get_cached_children().unwrap_or_default()
        } else {
            Vec::new()
        };
        if let Some(mut described) = describe(&element, elements) {
            if level < depth {
                described.child_count = Some(below.len());
            }
            out.push((level, described));
        }
        stack.extend(below.into_iter().rev().map(|e| (level + 1, e)));
    }
    Ok(out)
}

fn windows(
    automation: &UIAutomation,
    elements: &mut HashMap<String, UIElement>,
) -> Result<Vec<WindowEntry>, PlatformError> {
    let root = automation.get_root_element().map_err(failed)?;
    let walker = automation.get_control_view_walker().map_err(failed)?;
    let front = super::active_window().map(|(w, _)| w.process_id);
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let mut out = Vec::new();
    let mut next = walker.get_first_child(&root).ok();
    while let Some(window) = next {
        next = walker.get_next_sibling(&window).ok();
        let title = window.get_name().unwrap_or_default();
        let Ok(pid) = window.get_process_id() else {
            continue;
        };
        if title.is_empty() || pid == std::process::id() {
            continue;
        }
        let app = system
            .process(sysinfo::Pid::from_u32(pid))
            .map(|p| p.name().to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(id) = element_id(&window) else {
            continue;
        };
        elements.insert(id.clone(), window.clone());
        out.push(WindowEntry {
            id,
            app,
            title,
            front: front == Some(u64::from(pid)),
        });
    }
    Ok(out)
}

fn snapshot(
    automation: &UIAutomation,
    privacy: &Privacy,
    elements: &mut HashMap<String, UIElement>,
) -> Result<ContextSnapshot, PlatformError> {
    let mut snapshot = window_snapshot()?;
    match automation.get_focused_element() {
        Ok(element) => {
            let max = privacy.max_context_chars as i32;
            let (mut focused, errors) = read_element(&element, max);
            snapshot.errors.extend(errors);
            // Remembered by its id, so the inspector can walk up to it from the window.
            focused.id = element_id(&element);
            if let Some(id) = &focused.id {
                elements.insert(id.clone(), element.clone());
            }
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
        // A recording must not take the app's own typing for the user's.
        crate::hold::own_input();
        let _mark = OwnInput;
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
                    reason: "the flow delivers to the clipboard".into(),
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
