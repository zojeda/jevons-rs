//! `jevons-desktop.toml`: the runtime, models, dictation and privacy settings. The Settings
//! panel edits and saves it; a missing file means defaults.

use crate::context::Privacy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfig {
    pub server: Server,
    pub models: Models,
    pub dictation: Dictation,
    pub privacy: Privacy,
    /// The profiles folder; defaults to `profiles` next to this file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiles_dir: Option<PathBuf>,
}

/// Where inference runs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Load the models in this process.
    #[default]
    Embedded,
    /// Use a jevons server elsewhere.
    Remote,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    pub mode: Mode,
    /// Embedded: serve the API to other clients on `bind:port`. When off, the API listens on
    /// an ephemeral loopback port with a key only this app knows.
    pub expose: bool,
    pub bind: IpAddr,
    pub port: u16,
    /// The key other clients use when exposed; `TYPESAFE_API_KEY` wins when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Remote: the server root.
    pub remote_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_key: Option<String>,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            mode: Mode::Embedded,
            expose: false,
            bind: IpAddr::from([127, 0, 0, 1]),
            port: 8080,
            api_key: None,
            remote_url: "http://127.0.0.1:8080".into(),
            remote_key: None,
        }
    }
}

/// A model on disk.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    /// A GGUF file or a Hugging Face checkpoint directory.
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<PathBuf>,
    /// The catalog entry it was downloaded from, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Models {
    /// Where downloads go; defaults to the platform data folder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folder: Option<PathBuf>,
    /// An existing jevons-rs settings file to load instead of the selections below.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_config: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generative: Option<ModelRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<ModelRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech: Option<ModelRef>,
    /// Serve Realtime transcription (live text while speaking).
    pub realtime: bool,
}

impl Default for Models {
    fn default() -> Self {
        Self {
            folder: None,
            runtime_config: None,
            generative: None,
            decision: None,
            speech: None,
            realtime: true,
        }
    }
}

impl Models {
    /// These selections with each unset service filled from the first downloaded catalog entry
    /// that serves it, in catalog order (DiffusionGemma first for language, then Parakeet for
    /// speech). Explicit selections are kept.
    pub fn with_defaults(&self, folder: &Path, catalog: &[crate::catalog::CatalogEntry]) -> Self {
        use crate::catalog::Service;
        let pick = |service: Service| {
            catalog
                .iter()
                .find(|e| e.serves(service) && e.is_ready(folder))
                .map(|e| ModelRef {
                    path: e.model_path(folder),
                    mmproj: e.mmproj_path(folder),
                    catalog: Some(e.id.clone()),
                })
        };
        let mut models = self.clone();
        if models.runtime_config.is_none() {
            models.generative = models.generative.or_else(|| pick(Service::Generative));
            models.decision = models.decision.or_else(|| pick(Service::Decision));
            models.speech = models.speech.or_else(|| pick(Service::Speech));
        }
        models
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dictation {
    /// Push-to-talk, and the per-profile hotkeys, which work the same way.
    pub hotkey: String,
    /// Whether the push-to-talk hotkeys are held while speaking or pressed to start and stop.
    pub hotkey_mode: HotkeyMode,
    /// Live dictation (F9 by default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_hotkey: Option<String>,
    pub live_hotkey_mode: HotkeyMode,
    /// A bubble by the tray icon shows what dictation hears and does: the words as they are
    /// recognized, the profile chosen, and the text delivered.
    pub live_feedback: bool,
    /// Ignored: live dictation used to type while speaking.
    #[serde(rename = "live_stream", skip_serializing)]
    pub legacy_live_stream: Option<bool>,
    /// Shows the inspector window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inspector_hotkey: Option<String>,
    /// Dictate with a given profile, whatever the context matches: profile id → hotkey.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub profile_hotkeys: BTreeMap<String, String>,
    /// The capture device name; the default device when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub microphone: Option<String>,
    /// An ISO-639-1 code; detected when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Ask the decision model which action to take and whether to rewrite.
    pub decide: bool,
    /// Below this probability that the text needs editing, the transcript is typed as is.
    pub generation_threshold: f64,
    /// The most tokens a rewrite may generate.
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
            legacy_live_stream: None,
            inspector_hotkey: None,
            profile_hotkeys: BTreeMap::new(),
            microphone: None,
            language: None,
            decide: true,
            generation_threshold: 0.5,
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

impl DesktopConfig {
    /// Reads `file`; a missing file gives the defaults.
    pub fn load(file: &Path) -> Result<Self, ConfigError> {
        match std::fs::read_to_string(file) {
            Ok(text) => toml::from_str(&text).map_err(|e| ConfigError::Invalid {
                file: file.into(),
                message: e.to_string(),
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Io {
                file: file.into(),
                source,
            }),
        }
    }

    /// Writes `file`, creating its folder.
    pub fn save(&self, file: &Path) -> Result<(), ConfigError> {
        let io = |source| ConfigError::Io {
            file: file.into(),
            source,
        };
        if let Some(dir) = file.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let text = toml::to_string_pretty(self).expect("the settings serialize");
        std::fs::write(file, text).map_err(io)
    }

    pub fn profiles_dir(&self, config_file: &Path) -> PathBuf {
        self.profiles_dir.clone().unwrap_or_else(|| {
            config_file
                .parent()
                .unwrap_or(Path::new("."))
                .join("profiles")
        })
    }

    /// The models folder: the configured one, else `~/jevons/models`.
    pub fn models_folder(&self) -> PathBuf {
        self.models
            .folder
            .clone()
            .unwrap_or_else(|| user_dir().join("models"))
    }

    /// The key for an exposed embedded API: `TYPESAFE_API_KEY`, then the settings.
    pub fn exposed_key(&self) -> Option<String> {
        std::env::var("TYPESAFE_API_KEY")
            .ok()
            .or_else(|| self.server.api_key.clone())
            .filter(|k| !k.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_settings_file_parses() {
        let text = include_str!("../../../jevons-desktop.example.toml");
        let config: DesktopConfig = toml::from_str(text).unwrap();
        assert!(!config.server.expose);
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn hotkey_modes_are_hold_or_toggle() {
        let config: DesktopConfig =
            toml::from_str("[dictation]\nlive_hotkey_mode = \"toggle\"\n").unwrap();
        assert_eq!(config.dictation.hotkey_mode, HotkeyMode::Hold);
        assert_eq!(config.dictation.live_hotkey_mode, HotkeyMode::Toggle);
        assert!(toml::from_str::<DesktopConfig>("[dictation]\nhotkey_mode = \"tap\"\n").is_err());
    }

    #[test]
    fn settings_saved_before_live_feedback_still_load() {
        let config: DesktopConfig = toml::from_str("[dictation]\nlive_stream = true\n").unwrap();
        assert!(config.dictation.live_feedback);
        let saved = toml::to_string(&config).unwrap();
        assert!(!saved.contains("live_stream"), "{saved}");
    }

    #[test]
    fn saved_settings_load_back_unchanged() {
        let dir =
            std::env::temp_dir().join(format!("jevons-desktop-config-{}", std::process::id()));
        let file = dir.join("jevons-desktop.toml");
        let mut config = DesktopConfig::default();
        config.server.expose = true;
        config.server.port = 8081;
        config.dictation.inspector_hotkey = Some("Ctrl+Alt+I".into());
        config
            .dictation
            .profile_hotkeys
            .insert("chat".into(), "Ctrl+Alt+C".into());
        config.models.speech = Some(ModelRef {
            path: "/models/parakeet".into(),
            mmproj: None,
            catalog: Some("parakeet-tdt-0.6b-v3".into()),
        });
        config.save(&file).unwrap();
        assert_eq!(DesktopConfig::load(&file).unwrap(), config);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn downloaded_catalog_models_fill_unset_services_with_gemma_first() {
        let folder = std::env::temp_dir().join(format!("jevons-defaults-{}", std::process::id()));
        let catalog = crate::catalog::builtin();
        for id in [
            "diffusiongemma-26b-a4b-q4_k_m",
            "nemotron-labs-diffusion-3b",
            "parakeet-tdt-0.6b-v3",
        ] {
            let dir = folder.join(id);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(crate::download::COMPLETE_MARKER), b"").unwrap();
        }
        let chosen = Models {
            speech: Some(ModelRef {
                path: "/mine".into(),
                mmproj: None,
                catalog: None,
            }),
            ..Models::default()
        }
        .with_defaults(&folder, &catalog);
        let generative = chosen.generative.unwrap();
        assert_eq!(
            generative.catalog.as_deref(),
            Some("diffusiongemma-26b-a4b-q4_k_m")
        );
        assert!(
            generative
                .path
                .ends_with("diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
        );
        assert!(generative.mmproj.is_some());
        assert_eq!(chosen.decision.unwrap().catalog, generative.catalog);
        assert_eq!(
            chosen.speech.unwrap().path,
            Path::new("/mine"),
            "explicit choices stay"
        );
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn a_missing_file_gives_defaults_and_unknown_fields_are_errors() {
        let missing = std::env::temp_dir().join("jevons-desktop-missing.toml");
        assert_eq!(
            DesktopConfig::load(&missing).unwrap(),
            DesktopConfig::default()
        );
        assert!(toml::from_str::<DesktopConfig>("[server]\nprot = 1").is_err());
    }
}
