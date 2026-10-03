//! The platform layers, one trait each. Every desktop implements these; the behaviour around
//! them ([`pipeline`](crate::pipeline), the flow tree, gestures, tray states) is the same
//! everywhere.
//! [`Unsupported`] stands in for layers a platform does not have yet.

use crate::context::{ContextSnapshot, Privacy};
use crate::icons::TrayState;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

/// The rate of [`AudioEvent::Chunk`] samples: the Realtime API's default PCM16 rate.
pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("{0} is not supported on this platform yet")]
    Unsupported(&'static str),
    #[error("{0}")]
    Failed(String),
}

/// Accessibility layer: what the user is dictating into.
pub trait ContextProvider: Send {
    /// The backend, such as `UI Automation` or `AT-SPI`, for the inspector.
    fn name(&self) -> &'static str;
    /// The focused application and element now, sanitized with `privacy`.
    fn snapshot(&self, privacy: &Privacy) -> Result<ContextSnapshot, PlatformError>;
}

/// A top-level window the context investigator may look into.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct WindowEntry {
    /// Stable while the window exists.
    pub id: String,
    /// The process name, such as `slack.exe`.
    pub app: String,
    pub title: String,
    /// The window in front.
    pub front: bool,
}

/// An element of an application's interface, as the accessibility layer reports it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct UiElement {
    /// Stable within an investigation; used to read the element's children.
    pub id: String,
    /// The control type, such as `List`, `ListItem` or `Text`.
    pub role: String,
    pub name: String,
    /// The value or document text, when the element has one.
    pub value: Option<String>,
    pub class: Option<String>,
    pub automation_id: Option<String>,
    /// A password field: its text is never read.
    pub password: bool,
    /// How many children it has, when known.
    pub child_count: Option<usize>,
    /// Whether it takes input; `None` when the layer did not say.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Whether it is scrolled or hidden out of view.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offscreen: Option<bool>,
    /// A selectable item: whether it is selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected: Option<bool>,
    /// A toggle (check box, switch): whether it is on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toggled: Option<bool>,
    /// An expandable item (tree item, combo box): whether it is expanded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expanded: Option<bool>,
}

/// How far below an element a search looks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reach {
    Children,
    Descendants,
}

/// A property an accessibility layer matches natively.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Property {
    Role,
    Name,
    Class,
    AutomationId,
}

/// One condition of a native search: the property equals (or contains) the text. A search's
/// conditions all hold; a layer that cannot match one natively may ignore it, since callers
/// check every match again.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Condition {
    pub property: Property,
    pub value: String,
    pub substring: bool,
}

impl Condition {
    pub fn matches(&self, element: &UiElement) -> bool {
        let actual = match self.property {
            Property::Role => Some(element.role.as_str()),
            Property::Name => Some(element.name.as_str()),
            Property::Class => element.class.as_deref(),
            Property::AutomationId => element.automation_id.as_deref(),
        };
        actual.is_some_and(|actual| {
            if self.substring {
                actual.contains(&self.value)
            } else {
                actual == self.value
            }
        })
    }
}

/// Accessibility layer for investigations: windows and their element trees, read on demand.
pub trait ContextInspector: Send + Sync {
    /// The backend, for the inspector.
    fn name(&self) -> &'static str;
    fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError>;
    /// The children of a window (by [`WindowEntry::id`]) or of an element (by [`UiElement::id`]).
    fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError>;
    /// The element with the keyboard focus, and the top-level window it is in.
    fn focused(&self) -> Result<Option<(UiElement, WindowEntry)>, PlatformError> {
        Err(PlatformError::Unsupported("Reading the focused element"))
    }
    /// The parent of an element, `None` for a top-level window. Needed for `..` and the
    /// ancestor axes after a search, which does not say where its matches sit.
    fn parent(&self, _id: &str) -> Result<Option<UiElement>, PlatformError> {
        Err(PlatformError::Unsupported("Reading an element's parent"))
    }
    /// The elements below `id` that meet every condition, in document order, at most `limit`.
    /// Platforms override this with a native search; this walks.
    fn find(
        &self,
        id: &str,
        reach: Reach,
        conditions: &[Condition],
        limit: usize,
    ) -> Result<Vec<UiElement>, PlatformError> {
        let walked = match reach {
            Reach::Children => self.children(id)?,
            Reach::Descendants => self
                .subtree(id, 64, 20_000)?
                .into_iter()
                .map(|(_, e)| e)
                .collect(),
        };
        Ok(walked
            .into_iter()
            .filter(|e| conditions.iter().all(|c| c.matches(e)))
            .take(limit)
            .collect())
    }
    /// Every element below `id` to `depth`, parents before children, at most `limit` of them.
    /// Platforms override this to read a subtree in one call.
    fn subtree(
        &self,
        id: &str,
        depth: usize,
        limit: usize,
    ) -> Result<Vec<(usize, UiElement)>, PlatformError> {
        let mut out = Vec::new();
        let mut stack: Vec<(usize, UiElement)> = self
            .children(id)?
            .into_iter()
            .rev()
            .map(|e| (1, e))
            .collect();
        while let Some((level, element)) = stack.pop() {
            if out.len() >= limit {
                break;
            }
            if level < depth && element.child_count != Some(0) {
                let children = self.children(&element.id).unwrap_or_default();
                stack.extend(children.into_iter().rev().map(|e| (level + 1, e)));
            }
            out.push((level, element));
        }
        Ok(out)
    }
}

/// A capture device.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AudioDevice {
    pub name: String,
    pub default: bool,
}

/// What an [`AudioSource`] reports while capturing.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioEvent {
    /// Mono PCM16 at [`SAMPLE_RATE`], about 100 ms each.
    Chunk(Vec<i16>),
    /// The meter bands for the latest audio, 0..=16.
    Level([u8; 5]),
    /// The source ended by itself (a file finished, a device went away).
    Ended,
    Failed(String),
}

/// Stops a capture when dropped or stopped.
pub trait CaptureHandle: Send {
    fn stop(self: Box<Self>);
}

/// Audio capture layer.
pub trait AudioSource: Send {
    fn devices(&self) -> Vec<AudioDevice>;
    /// Starts capturing from `device` (the default when `None`) into `events`.
    fn start(
        &mut self,
        device: Option<&str>,
        events: UnboundedSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureHandle>, PlatformError>;
}

/// What to do with the text in the target.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Insert at the caret.
    #[default]
    Insert,
    /// Replace the selection.
    Replace,
    /// Replace the selection, or the whole field when nothing is selected.
    Rewrite,
}

/// How the text reaches the target.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMethod {
    /// Put the text on the clipboard and paste it, restoring the clipboard after.
    #[default]
    Paste,
    /// Type it key by key.
    Type,
    /// Set the element's value through the accessibility API.
    SetValue,
    /// Only copy it; the user pastes.
    Clipboard,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeliveryRequest {
    pub action: Action,
    pub text: String,
    pub method: DeliveryMethod,
    /// Whether the field's text must be selected first (a rewrite with nothing selected).
    pub select_all: bool,
    /// Characters to delete before the caret first (live dictation correcting what it typed).
    pub erase: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum DeliveryOutcome {
    Delivered {
        method: DeliveryMethod,
    },
    /// The text is on the clipboard for the user to paste.
    OnClipboard {
        reason: String,
    },
    /// Shown in the feedback bubble, where the user can copy or insert it.
    Shown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SinkCapabilities {
    pub paste: bool,
    pub type_text: bool,
    pub set_value: bool,
}

/// Text input layer.
pub trait TextSink: Send {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> SinkCapabilities;
    /// The foreground window handle, compared with the one the take started in.
    fn foreground_window(&self) -> Option<u64>;
    /// Whether any key is held (delivery waits so held modifiers do not change the paste).
    fn keys_down(&self) -> bool;
    fn deliver(&mut self, request: &DeliveryRequest) -> Result<DeliveryOutcome, PlatformError>;
    /// Puts `text` on the clipboard.
    fn copy(&mut self, text: &str) -> Result<(), PlatformError>;
}

/// What a global hotkey does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotkeyAction {
    /// Push-to-talk: listening while held, the take runs on release. `entry` starts the take at
    /// that branch of the flow tree instead of its root.
    Dictate {
        entry: Option<String>,
    },
    /// Live dictation while held (from the tray, a start/stop toggle): words are typed as they
    /// are recognized.
    LiveDictation,
    ShowInspector,
    /// Answers the tool call the bubble asks about (Enter runs it, Esc cancels); registered only
    /// while a call waits.
    Confirm(bool),
    /// Starts recording a demonstration, or, while one records: held, a spoken note (the first
    /// one says what the task is); tapped, the end of the recording.
    Record,
    /// Runs this automation; when it takes arguments, the user says them while holding.
    Automation {
        name: String,
    },
}

/// A global shortcut, such as `Ctrl+Alt+Space`, and what it does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub accelerator: String,
    pub action: HotkeyAction,
}

impl Binding {
    /// Every hotkey the dictation settings define.
    pub fn from_settings(dictation: &crate::config::Dictation) -> Vec<Self> {
        let mut bindings = vec![Self {
            accelerator: dictation.hotkey.clone(),
            action: HotkeyAction::Dictate { entry: None },
        }];
        if let Some(accelerator) = dictation.live_hotkey.clone().filter(|a| !a.is_empty()) {
            bindings.push(Self {
                accelerator,
                action: HotkeyAction::LiveDictation,
            });
        }
        if let Some(accelerator) = dictation.inspector_hotkey.clone().filter(|a| !a.is_empty()) {
            bindings.push(Self {
                accelerator,
                action: HotkeyAction::ShowInspector,
            });
        }
        for (entry, accelerator) in &dictation.branch_hotkeys {
            if !accelerator.is_empty() {
                bindings.push(Self {
                    accelerator: accelerator.clone(),
                    action: HotkeyAction::Dictate {
                        entry: Some(entry.clone()),
                    },
                });
            }
        }
        bindings
    }

    /// The recording hotkey and a hotkey per automation.
    pub fn for_automations(settings: &crate::config::AutomationSettings) -> Vec<Self> {
        let mut bindings = Vec::new();
        if let Some(accelerator) = settings.record_hotkey.clone().filter(|a| !a.is_empty()) {
            bindings.push(Self {
                accelerator,
                action: HotkeyAction::Record,
            });
        }
        for (name, accelerator) in &settings.hotkeys {
            if !accelerator.is_empty() {
                bindings.push(Self {
                    accelerator: accelerator.clone(),
                    action: HotkeyAction::Automation { name: name.clone() },
                });
            }
        }
        bindings
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotkeyEvent {
    Pressed(u32),
    Released(u32),
}

/// The tray menu's contents.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MenuModel {
    pub dictating: bool,
    /// Whether a take is running (and can be cancelled).
    pub busy: bool,
    /// Whether live dictation runs.
    pub live: bool,
    /// The flow tree's top-level branches (path and description), where a take may start.
    pub entries: Vec<(String, String)>,
    /// The branch every take starts at, chosen from the menu; `None` starts at the root.
    pub start: Option<String>,
    pub context_paused: bool,
    /// Whether the feedback bubble shows.
    pub feedback: bool,
    /// Whether a demonstration is being recorded.
    pub recording: bool,
    /// The automations library: name, description, and whether this version is approved.
    pub automations: Vec<(String, String, bool)>,
    /// Whether a task waits with a conversation the bubble can show again.
    pub conversation: bool,
}

/// What a tray menu item asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuCommand {
    ToggleDictation,
    ToggleLiveDictation,
    /// Abandons the running take without delivering anything.
    CancelTake,
    /// Shows the waiting task's conversation in the bubble again.
    ShowConversation,
    /// Opens the folder with the logs and take traces.
    OpenLogsFolder,
    /// Starts every take at this top-level branch, or at the root.
    StartAt(Option<String>),
    ShowInspector,
    ToggleContextPause,
    /// Shows or hides the feedback bubble by the tray icon.
    ToggleFeedback,
    ReloadFlows,
    OpenConfigFolder,
    /// Asks, then puts the default settings, flow tree and automations library back.
    ResetSettings,
    /// Asks, then clears these kinds of history.
    ClearHistory(Vec<crate::history::History>),
    /// Starts or stops recording a demonstration.
    ToggleRecording,
    RunAutomation(String),
    /// Runs an automation asking before each of its actions.
    RunStepByStep(String),
    /// Records the automation's task again, to replace it with a new version.
    RecordAgain(String),
    /// Checks an automation, shows what it does, and pins this version when the user agrees.
    ApproveAutomation(String),
    OpenAutomationsFolder,
    Quit,
}

/// Desktop menu layer.
pub trait TrayBackend {
    fn set_state(&mut self, state: TrayState);
    fn set_menu(&mut self, menu: &MenuModel);
}

/// Something to do to an element an inspector found.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "action", content = "text", rename_all = "snake_case")]
pub enum UiAction {
    /// Its own action through the accessibility API (a button's press, a link's follow), or a
    /// click when it has none.
    Invoke,
    /// A mouse click at its clickable point.
    Click,
    /// Moves the keyboard focus to it.
    Focus,
    /// Replaces its value through the accessibility API.
    SetValue(String),
    /// Focuses it and types the text key by key.
    TypeText(String),
    Toggle,
    Select,
    Expand,
    Collapse,
    /// Scrolls its container until it is in view.
    ScrollIntoView,
}

impl UiAction {
    /// The action's name, as scripts and traces write it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Invoke => "invoke",
            Self::Click => "click",
            Self::Focus => "focus",
            Self::SetValue(_) => "set_value",
            Self::TypeText(_) => "type_text",
            Self::Toggle => "toggle",
            Self::Select => "select",
            Self::Expand => "expand",
            Self::Collapse => "collapse",
            Self::ScrollIntoView => "scroll_into_view",
        }
    }

    /// The text it enters, for the actions that enter text.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::SetValue(text) | Self::TypeText(text) => Some(text),
            _ => None,
        }
    }
}

/// How an action was carried out, for the trace.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Acted {
    /// Such as "the invoke pattern" or "a click at 120, 340".
    pub how: String,
}

/// A modifier key of a [`Chord`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Modifier {
    Ctrl,
    Alt,
    Shift,
    /// The Windows or Command key.
    Meta,
}

/// The key of a [`Chord`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Key {
    /// A printable character, such as `k` or `/`.
    Char(char),
    Enter,
    Tab,
    Escape,
    Backspace,
    Delete,
    Insert,
    Space,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    /// F1 to F24.
    Function(u8),
}

/// Keys pressed together, such as `ctrl+k`, `shift+tab` or `enter`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Chord {
    pub modifiers: Vec<Modifier>,
    pub key: Key,
}

impl Chord {
    /// Reads `ctrl+shift+k`: modifiers, then one key, joined with `+` (ignoring case).
    pub fn parse(text: &str) -> Result<Self, String> {
        let parts: Vec<String> = text.split('+').map(|p| p.trim().to_lowercase()).collect();
        let Some((key, modifiers)) = parts.split_last() else {
            return Err("an empty chord".into());
        };
        let mut out = Vec::new();
        for modifier in modifiers {
            let modifier = match modifier.as_str() {
                "ctrl" | "control" => Modifier::Ctrl,
                "alt" | "option" => Modifier::Alt,
                "shift" => Modifier::Shift,
                "meta" | "win" | "cmd" | "command" | "super" => Modifier::Meta,
                other => {
                    return Err(format!(
                        "{other:?} is not a modifier: use ctrl, alt, shift or meta, then one key"
                    ));
                }
            };
            if out.contains(&modifier) {
                return Err(format!("{text:?} repeats a modifier"));
            }
            out.push(modifier);
        }
        out.sort();
        let key = match key.as_str() {
            "enter" | "return" => Key::Enter,
            "tab" => Key::Tab,
            "escape" | "esc" => Key::Escape,
            "backspace" => Key::Backspace,
            "delete" | "del" => Key::Delete,
            "insert" => Key::Insert,
            "space" => Key::Space,
            "up" => Key::Up,
            "down" => Key::Down,
            "left" => Key::Left,
            "right" => Key::Right,
            "home" => Key::Home,
            "end" => Key::End,
            "pageup" => Key::PageUp,
            "pagedown" => Key::PageDown,
            f if f.len() > 1
                && f.starts_with('f')
                && f[1..].parse::<u8>().is_ok_and(|n| (1..=24).contains(&n)) =>
            {
                Key::Function(f[1..].parse().expect("checked"))
            }
            single if single.chars().count() == 1 => {
                let c = single.chars().next().expect("one character");
                if c.is_control() || c.is_whitespace() {
                    return Err(format!("{text:?}: write space, tab or enter by name"));
                }
                Key::Char(c)
            }
            "" => return Err(format!("{text:?} has no key after its modifiers")),
            other => {
                return Err(format!(
                    "{other:?} is not a key: use one character, enter, tab, escape, backspace, \
                     delete, space, up, down, left, right, home, end, pageup, pagedown or f1 to f24"
                ));
            }
        };
        Ok(Self {
            modifiers: out,
            key,
        })
    }
}

impl std::fmt::Display for Chord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for modifier in &self.modifiers {
            let name = match modifier {
                Modifier::Ctrl => "ctrl",
                Modifier::Alt => "alt",
                Modifier::Shift => "shift",
                Modifier::Meta => "meta",
            };
            write!(f, "{name}+")?;
        }
        match &self.key {
            Key::Char(c) => write!(f, "{c}"),
            Key::Function(n) => write!(f, "f{n}"),
            other => {
                let name = serde_json::to_value(other)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_default();
                f.write_str(&name.replace('_', ""))
            }
        }
    }
}

/// What the user did, as a platform's recorder saw it.
#[derive(Clone, Debug, PartialEq)]
pub enum Observed {
    /// A left click on this element, in this top-level window.
    Click {
        element: Box<UiElement>,
        window: WindowEntry,
    },
    /// Keys pressed together: any with ctrl, alt or meta, and named keys such as enter, tab or
    /// backspace.
    Chord(Chord),
    /// A printable character typed (shift and the keyboard layout applied).
    Char(char),
}

/// Stops a recording when stopped or dropped.
pub trait RecordingHandle: Send {
    fn stop(self: Box<Self>);
}

/// Recording layer: what the user does with the mouse and the keyboard, while a
/// demonstration is recorded. Input the app sends itself is not reported.
pub trait Recorder: Send + Sync {
    fn name(&self) -> &'static str;
    fn start(
        &self,
        events: UnboundedSender<Observed>,
    ) -> Result<Box<dyn RecordingHandle>, PlatformError>;
}

/// Action layer: acts on elements a [`ContextInspector`] returned (by [`UiElement::id`]), and
/// sends keys and text to the window in front. Callers check what may be acted on first (see
/// `automation::hands`); this layer only carries it out.
pub trait UiActor: Send + Sync {
    /// The backend, for traces.
    fn name(&self) -> &'static str;
    fn act(&self, id: &str, action: &UiAction) -> Result<Acted, PlatformError>;
    /// Presses a chord in the window in front.
    fn press(&self, chord: &Chord) -> Result<(), PlatformError>;
    /// Types text into the window in front.
    fn type_text(&self, text: &str) -> Result<(), PlatformError>;
    /// Brings a top-level window (by [`WindowEntry::id`]) to the front.
    fn activate(&self, window: &str) -> Result<(), PlatformError>;
    /// The process name of the window in front, such as `slack.exe`.
    fn front_app(&self) -> Option<String>;
}

/// A layer this platform does not implement yet.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unsupported;

impl ContextProvider for Unsupported {
    fn name(&self) -> &'static str {
        "none"
    }

    fn snapshot(&self, _: &Privacy) -> Result<ContextSnapshot, PlatformError> {
        Err(PlatformError::Unsupported(
            "Reading the focused application",
        ))
    }
}

impl ContextInspector for Unsupported {
    fn name(&self) -> &'static str {
        "none"
    }

    fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
        Err(PlatformError::Unsupported(
            "Reading other applications' interfaces",
        ))
    }

    fn children(&self, _: &str) -> Result<Vec<UiElement>, PlatformError> {
        Err(PlatformError::Unsupported(
            "Reading other applications' interfaces",
        ))
    }
}

impl TextSink for Unsupported {
    fn name(&self) -> &'static str {
        "none"
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities::default()
    }

    fn foreground_window(&self) -> Option<u64> {
        None
    }

    fn keys_down(&self) -> bool {
        false
    }

    fn deliver(&mut self, _: &DeliveryRequest) -> Result<DeliveryOutcome, PlatformError> {
        Err(PlatformError::Unsupported("Typing into other applications"))
    }

    fn copy(&mut self, _: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported("The clipboard"))
    }
}

impl UiActor for Unsupported {
    fn name(&self) -> &'static str {
        "none"
    }

    fn act(&self, _: &str, _: &UiAction) -> Result<Acted, PlatformError> {
        Err(PlatformError::Unsupported("Acting on other applications"))
    }

    fn press(&self, _: &Chord) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported(
            "Pressing keys in other applications",
        ))
    }

    fn type_text(&self, _: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported("Typing into other applications"))
    }

    fn activate(&self, _: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported("Bringing windows to the front"))
    }

    fn front_app(&self) -> Option<String> {
        None
    }
}

impl Recorder for Unsupported {
    fn name(&self) -> &'static str {
        "none"
    }

    fn start(
        &self,
        _: UnboundedSender<Observed>,
    ) -> Result<Box<dyn RecordingHandle>, PlatformError> {
        Err(PlatformError::Unsupported("Recording demonstrations"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_record_and_automation_hotkeys_bind_from_the_settings() {
        let mut settings = crate::config::AutomationSettings {
            record_hotkey: Some("Ctrl+Alt+R".into()),
            ..crate::config::AutomationSettings::default()
        };
        settings
            .hotkeys
            .insert("slack-post".into(), "Ctrl+Alt+P".into());
        settings.hotkeys.insert("unbound".into(), String::new());
        assert_eq!(
            Binding::for_automations(&settings),
            [
                Binding {
                    accelerator: "Ctrl+Alt+R".into(),
                    action: HotkeyAction::Record
                },
                Binding {
                    accelerator: "Ctrl+Alt+P".into(),
                    action: HotkeyAction::Automation {
                        name: "slack-post".into()
                    }
                },
            ]
        );
        assert!(Binding::for_automations(&crate::config::AutomationSettings::default()).is_empty());
    }

    #[test]
    fn chords_read_modifiers_then_one_key_and_write_back_the_same() {
        let chord = Chord::parse("Shift+Ctrl+K").unwrap();
        assert_eq!(chord.modifiers, [Modifier::Ctrl, Modifier::Shift]);
        assert_eq!(chord.key, Key::Char('k'));
        assert_eq!(chord.to_string(), "ctrl+shift+k");
        assert_eq!(Chord::parse("enter").unwrap().key, Key::Enter);
        assert_eq!(Chord::parse("alt+f4").unwrap().to_string(), "alt+f4");
        assert_eq!(Chord::parse("pagedown").unwrap().to_string(), "pagedown");
        assert_eq!(Chord::parse("ctrl+/").unwrap().key, Key::Char('/'));
        assert!(
            Chord::parse("hyper+k")
                .unwrap_err()
                .contains("not a modifier")
        );
        assert!(Chord::parse("ctrl+").unwrap_err().contains("no key"));
        assert!(Chord::parse("ctrl+ctrl+k").unwrap_err().contains("repeats"));
        assert!(
            Chord::parse("ctrl+enterr")
                .unwrap_err()
                .contains("not a key")
        );
        assert!(Chord::parse("f25").is_err());
    }
}
