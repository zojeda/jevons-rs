//! `jevons-desktop.toml`: the client's settings. Dictation and its hotkeys, what a snapshot may
//! keep, the tools that run on this machine, and the automations library with its approvals. The Settings panel edits and saves it;
//! a missing file means defaults.

use crate::context::Privacy;
pub use jevons_desktop_tools::config::{McpConfig, ToolConfig, ToolKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClientConfig {
    pub dictation: Dictation,
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
}

/// `[automation]`: the automations library and the versions the user approved.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutomationSettings {
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
