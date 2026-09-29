//! The platform layers, one trait each. Every desktop implements these; the behaviour around
//! them ([`pipeline`](crate::pipeline), profiles, gestures, tray states) is the same everywhere.
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
    /// Push-to-talk: listening while held, the take runs on release. `profile` forces a
    /// profile for the take.
    Dictate {
        profile: Option<String>,
    },
    /// Live dictation while held (from the tray, a start/stop toggle): words are typed as they
    /// are recognized.
    LiveDictation,
    ShowInspector,
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
            action: HotkeyAction::Dictate { profile: None },
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
        for (profile, accelerator) in &dictation.profile_hotkeys {
            if !accelerator.is_empty() {
                bindings.push(Self {
                    accelerator: accelerator.clone(),
                    action: HotkeyAction::Dictate {
                        profile: Some(profile.clone()),
                    },
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
    /// Profile ids and names, in display order.
    pub profiles: Vec<(String, String)>,
    /// The profile forced from the menu; `None` chooses automatically.
    pub forced: Option<String>,
    pub context_paused: bool,
    /// Whether the feedback bubble shows.
    pub feedback: bool,
}

/// What a tray menu item asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuCommand {
    ToggleDictation,
    ToggleLiveDictation,
    /// Abandons the running take without delivering anything.
    CancelTake,
    /// Opens the folder with the logs and take traces.
    OpenLogsFolder,
    ForceProfile(Option<String>),
    ShowInspector,
    ToggleContextPause,
    /// Shows or hides the feedback bubble by the tray icon.
    ToggleFeedback,
    ReloadProfiles,
    OpenConfigFolder,
    Quit,
}

/// Desktop menu layer.
pub trait TrayBackend {
    fn set_state(&mut self, state: TrayState);
    fn set_menu(&mut self, menu: &MenuModel);
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
