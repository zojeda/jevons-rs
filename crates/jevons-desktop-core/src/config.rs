//! `jevons-desktop.toml`: the client's settings. Dictation and its hotkeys, the voice that reads
//! answers aloud, what a snapshot may keep, the tools that run on this machine, and the
//! automations library with its approvals. The Settings panel edits and saves it; a missing
//! file means defaults.

use crate::context::Privacy;
pub use crate::control::ControlSettings;
pub use jevons_desktop_tools::config::{McpConfig, ToolConfig, ToolKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClientConfig {
    pub dictation: Dictation,
    /// Whether answers are read aloud, and on which device.
    #[serde(skip_serializing_if = "VoiceSettings::is_default")]
    pub voice: VoiceSettings,
    pub privacy: Privacy,
    /// Built-in tools that run on this machine, by name: flow nodes may call them, and each
    /// asks here before it runs unless it says otherwise.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, ToolConfig>,
    /// MCP servers that run on this machine, whose tools flow nodes may call as `server:tool`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, McpConfig>,
    /// The automations library, recording demonstrations, and which versions may run.
    #[serde(skip_serializing_if = "AutomationSettings::is_default")]
    pub automation: AutomationSettings,
    /// What an outside agent may do at this desk through the server's control API.
    #[serde(skip_serializing_if = "ControlSettings::is_default")]
    pub control: ControlSettings,
}

/// `[voice]`: whether jevons reads answers aloud, and where.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceSettings {
    /// Read aloud the answers of the flow leaves that ask for it (`output = "voice"`). Off,
    /// they are shown in the bubble and nothing is said.
    pub speak: bool,
    /// The output device name; the default device when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            speak: true,
            output: None,
        }
    }
}

impl VoiceSettings {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// `[automation]`: the automations library and the versions the user approved.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutomationSettings {
    /// Automations are coming soon: until this is set, which is how to try them, the app
    /// records, lists and runs none.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub enabled: bool,
    /// The library folder; `automations` next to this file by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    /// Where recorded demonstrations go; `~/jevons/recordings` by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recordings_dir: Option<PathBuf>,
    /// Starts and stops recording a demonstration. Held while recording, it records a spoken
    /// note.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_hotkey: Option<String>,
    /// A hotkey per automation, by name, such as `slack-post = "Ctrl+Alt+P"`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub hotkeys: BTreeMap<String, String>,
    /// The version (`sha256:…`) of each automation the user approved. Only the app writes it,
    /// when the user approves; an automation whose files changed since needs approving again.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub approved: BTreeMap<String, String>,
    /// Automations that run without asking in the bubble first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unconfirmed: Vec<String>,
    /// Globs on the flow nodes that may call each automation, by name; unlisted ones allow all.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub allow: BTreeMap<String, Vec<String>>,
    /// The model the built-in author writes scripts with; the generative model by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_model: Option<String>,
}

impl AutomationSettings {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// How a dictation hotkey starts and stops listening.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HotkeyMode {
    /// Listens while the hotkey is held; releasing it stops.
    #[default]
    Hold,
    /// The first press starts listening, the next press stops.
    Toggle,
}

/// An optional setting whose default is on, written as `""` when off: TOML has no null, and a
/// left-out key would read back as the default.
mod empty_is_none {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Option<String>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(value.as_deref().unwrap_or(""))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        Ok(Some(String::deserialize(d)?).filter(|s| !s.trim().is_empty()))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dictation {
    /// Push-to-talk, and the branch hotkeys, which work the same way.
    pub hotkey: String,
    /// Whether the push-to-talk hotkeys are held while speaking or pressed to start and stop.
    pub hotkey_mode: HotkeyMode,
    /// Live dictation (F9 by default); `""` turns it off.
    #[serde(with = "empty_is_none")]
    pub live_hotkey: Option<String>,
    pub live_hotkey_mode: HotkeyMode,
    /// A double-tap on Bluetooth headphones starts live dictation, which ends after a quiet
    /// pause; a single tap goes on to the last music app. Windows only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub headphones: bool,
    /// A word that starts live dictation when said alone, such as `Jevons`; off when unset.
    /// It ends after a quiet pause, as a take the headphones start does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wake_word: Option<String>,
    /// How the speech model wrote the wake word when the user taught it. Only the app writes it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub wake_spellings: Vec<String>,
    /// A bubble by the tray icon shows what dictation hears and does: the words as they are
    /// recognized, the route through the flow tree, and the text delivered.
    pub live_feedback: bool,
    /// Shows the inspector window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inspector_hotkey: Option<String>,
    /// Push-to-talk hotkeys that start the take at a branch of the flow tree instead of its
    /// root: branch path (such as `ask`) → hotkey.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub branch_hotkeys: BTreeMap<String, String>,
    /// The capture device name; the default device when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub microphone: Option<String>,
    /// An ISO-639-1 code; detected when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Ask the decision model at the flow tree's model decisions; when off, they take their
    /// fallback (or the best-ranked branch).
    pub decide: bool,
    /// The most tokens a generation writes, unless a flow node sets its own.
    pub max_output_tokens: u32,
}

impl Default for Dictation {
    fn default() -> Self {
        Self {
            hotkey: "Ctrl+Alt+Space".into(),
            hotkey_mode: HotkeyMode::Hold,
            live_hotkey: Some("F9".into()),
            live_hotkey_mode: HotkeyMode::Hold,
            headphones: false,
            wake_word: None,
            wake_spellings: Vec::new(),
            live_feedback: true,
            inspector_hotkey: None,
            branch_hotkeys: BTreeMap::new(),
            microphone: None,
            language: None,
            decide: true,
            max_output_tokens: 1024,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{file}: {message}")]
    Invalid { file: PathBuf, message: String },
    #[error("{file}: {source}")]
    Io {
        file: PathBuf,
        source: std::io::Error,
    },
}

/// Platform folders for configuration and data.
pub fn project_dirs() -> Option<directories::ProjectDirs> {
    directories::ProjectDirs::from("", "", "jevons")
}

/// `~/jevons`: the logs and traces, in a folder the user can find and review.
pub fn user_dir() -> PathBuf {
    directories::UserDirs::new()
        .map(|d| d.home_dir().join("jevons"))
        .unwrap_or_else(|| PathBuf::from("jevons"))
}

/// `jevons-desktop.toml` in the platform configuration folder.
pub fn default_config_file() -> PathBuf {
    project_dirs()
        .map(|d| d.config_dir().join("jevons-desktop.toml"))
        .unwrap_or_else(|| PathBuf::from("jevons-desktop.toml"))
}

impl ClientConfig {
    /// The automations library: the configured folder, else `automations` next to the settings.
    pub fn automations_dir(&self, config_file: &Path) -> PathBuf {
        self.automation.dir.clone().unwrap_or_else(|| {
            config_file
                .parent()
                .unwrap_or(Path::new("."))
                .join("automations")
        })
    }

    /// Where demonstrations are recorded: the configured folder, else `~/jevons/recordings`.
    pub fn recordings_dir(&self) -> PathBuf {
        self.automation
            .recordings_dir
            .clone()
            .unwrap_or_else(|| user_dir().join("recordings"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_settings_file_parses() {
        let text = include_str!("../../../jevons-desktop.example.toml");
        let config: ClientConfig = toml::from_str(text).unwrap();
        assert_eq!(config.dictation.hotkey, "Ctrl+Alt+Space");
        assert!(!config.privacy.read_clipboard);
        assert!(config.voice.speak);
        assert_eq!(config.voice.output, None);
    }

    #[test]
    fn the_voice_speaks_on_the_default_device_unless_the_settings_say_otherwise() {
        // Left out, answers are read aloud on the default device, and nothing of it is
        // written to a file that never set it.
        let default: ClientConfig = toml::from_str("").unwrap();
        assert_eq!(default.voice, VoiceSettings::default());
        assert!(default.voice.speak && default.voice.output.is_none());
        assert!(!toml::to_string(&default).unwrap().contains("voice"));
        // Muted, on a named device: saved, it loads back the same.
        let config: ClientConfig =
            toml::from_str("[voice]\nspeak = false\noutput = \"Headphones\"\n").unwrap();
        assert!(!config.voice.speak);
        assert_eq!(config.voice.output.as_deref(), Some("Headphones"));
        let back: ClientConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(back, config);
        // A key it does not have is an error, never ignored.
        assert!(toml::from_str::<ClientConfig>("[voice]\nvolume = 0.5\n").is_err());
    }

    #[test]
    fn hotkey_modes_are_hold_or_toggle() {
        let config: ClientConfig =
            toml::from_str("[dictation]\nlive_hotkey_mode = \"toggle\"\n").unwrap();
        assert_eq!(config.dictation.hotkey_mode, HotkeyMode::Hold);
        assert_eq!(config.dictation.live_hotkey_mode, HotkeyMode::Toggle);
        assert!(toml::from_str::<ClientConfig>("[dictation]\nhotkey_mode = \"tap\"\n").is_err());
    }

    #[test]
    fn the_headphones_are_off_until_the_settings_turn_them_on() {
        let default: ClientConfig = toml::from_str("").unwrap();
        assert!(!default.dictation.headphones);
        assert!(!toml::to_string(&default).unwrap().contains("headphones"));
        let on: ClientConfig = toml::from_str("[dictation]\nheadphones = true\n").unwrap();
        assert!(on.dictation.headphones);
        let back: ClientConfig = toml::from_str(&toml::to_string(&on).unwrap()).unwrap();
        assert_eq!(back, on);
    }

    #[test]
    fn the_wake_word_is_off_until_the_settings_name_one() {
        let default: ClientConfig = toml::from_str("").unwrap();
        assert_eq!(default.dictation.wake_word, None);
        assert!(default.dictation.wake_spellings.is_empty());
        let text = toml::to_string(&default).unwrap();
        assert!(!text.contains("wake"), "{text}");
        let on: ClientConfig =
            toml::from_str("[dictation]\nwake_word = \"Jevons\"\nwake_spellings = [\"Shabons\"]\n")
                .unwrap();
        assert_eq!(on.dictation.wake_word.as_deref(), Some("Jevons"));
        let back: ClientConfig = toml::from_str(&toml::to_string(&on).unwrap()).unwrap();
        assert_eq!(back, on);
    }

    #[test]
    fn a_live_hotkey_turned_off_stays_off_once_saved() {
        let mut config = ClientConfig::default();
        config.dictation.live_hotkey = None;
        let text = toml::to_string(&config).unwrap();
        assert!(text.contains("live_hotkey = \"\""), "{text}");
        let back: ClientConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.dictation.live_hotkey, None);
        // Left out, it is the default.
        let default: ClientConfig = toml::from_str("").unwrap();
        assert_eq!(default.dictation.live_hotkey.as_deref(), Some("F9"));
    }
}
