//! The app's settings: the client's (`jevons-desktop.toml`) and the server's
//! (`jevons-server.toml`, next to it), as the one value the app and its Settings panel work
//! with. Each half is read from and written to its own file.

use jevons_desktop_core::catalog::{CatalogEntry, Service};
pub use jevons_desktop_core::config::{
    AutomationSettings, ClientConfig, ConfigError, Dictation, HotkeyMode, default_config_file,
    user_dir,
};
use jevons_desktop_protocol::context::Privacy;
pub use jevons_desktop_server::config::{
    Capability, EMBEDDED, Logging, McpConfig, ModelRef, Models, Provider, ProviderKind, RouteTo,
    Routes, Server, ServerConfig, ToolConfig,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Starts or stops the API log, in `~/jevons/logs/api.log`, as the server's `privacy.log_api`
/// says.
pub fn log_api(on: bool) {
    use jevons_desktop_server::client::log;
    let file = on.then(|| user_dir().join("logs").join("api.log"));
    if let Some(file) = &file
        && !log::enabled()
    {
        tracing::info!(file = %file.display(), "Writing the API log");
    }
    log::set(file);
}

/// The server's settings file, next to the client's.
pub fn server_file(config_file: &Path) -> PathBuf {
    config_file.with_file_name("jevons-server.toml")
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfig {
    pub server: Server,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, Provider>,
    pub routes: Routes,
    pub models: Models,
    pub dictation: Dictation,
    pub privacy: Privacy,
    /// The server's `log_api`.
    pub log_api: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flows_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, ToolConfig>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, McpConfig>,
    pub automation: AutomationSettings,
}

fn read<T: DeserializeOwned + Default>(file: &Path) -> Result<T, ConfigError> {
    match std::fs::read_to_string(file) {
        Ok(text) => toml::from_str(&text).map_err(|e| ConfigError::Invalid {
            file: file.into(),
            message: e.to_string(),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(source) => Err(ConfigError::Io {
            file: file.into(),
            source,
        }),
    }
}

fn write<T: Serialize>(file: &Path, settings: &T) -> Result<(), ConfigError> {
    let io = |source| ConfigError::Io {
        file: file.into(),
        source,
    };
    if let Some(dir) = file.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    let text = toml::to_string_pretty(settings).expect("the settings serialize");
    std::fs::write(file, text).map_err(io)
}

impl DesktopConfig {
    /// The two halves as one.
    pub fn of(client: ClientConfig, server: ServerConfig) -> Self {
        Self {
            server: server.server,
            providers: server.providers,
            routes: server.routes,
            models: server.models,
            dictation: client.dictation,
            privacy: client.privacy,
            log_api: server.privacy.log_api,
            flows_dir: server.flows_dir,
            tools: server.tools,
            mcp: server.mcp,
            automation: client.automation,
        }
    }

    /// The client's half: what `jevons-desktop.toml` holds.
    pub fn client(&self) -> ClientConfig {
        ClientConfig {
            dictation: self.dictation.clone(),
            privacy: self.privacy.clone(),
            automation: self.automation.clone(),
        }
    }

    /// The server's half: what `jevons-server.toml` holds.
    pub fn server(&self) -> ServerConfig {
        ServerConfig {
            server: self.server.clone(),
            providers: self.providers.clone(),
            routes: self.routes.clone(),
            models: self.models.clone(),
            privacy: Logging {
                log_api: self.log_api,
            },
            flows_dir: self.flows_dir.clone(),
            tools: self.tools.clone(),
            mcp: self.mcp.clone(),
        }
    }

    /// Reads the client's settings from `file` and the server's from the file next to it; a
    /// missing file gives that half's defaults.
    pub fn load(file: &Path) -> Result<Self, ConfigError> {
        Ok(Self::of(read(file)?, read(&server_file(file))?))
    }

    /// Writes both files, creating their folder. The approvals are not the settings panel's to
    /// change: they are kept as the file has them.
    pub fn save(&self, file: &Path) -> Result<(), ConfigError> {
        let mut client = self.client();
        if let Ok(current) = read::<ClientConfig>(file) {
            client.automation.approved = current.automation.approved;
        }
        write(file, &client)?;
        write(&server_file(file), &self.server())
    }

    /// Pins `version` as the approved one of automation `name`, changing nothing else in the
    /// client's file; returns the settings as saved.
    pub fn approve(file: &Path, name: &str, version: &str) -> Result<Self, ConfigError> {
        let mut client: ClientConfig = read(file)?;
        client
            .automation
            .approved
            .insert(name.to_string(), version.to_string());
        write(file, &client)?;
        Self::load(file)
    }

    pub fn flows_dir(&self, config_file: &Path) -> PathBuf {
        self.server().flows_dir(config_file)
    }

    /// The automations library: the configured folder, else `automations` next to the settings.
    pub fn automations_dir(&self, config_file: &Path) -> PathBuf {
        self.client().automations_dir(config_file)
    }

    /// Where demonstrations are recorded: the configured folder, else `~/jevons/recordings`.
    pub fn recordings_dir(&self) -> PathBuf {
        self.client().recordings_dir()
    }

    /// The models folder: the configured one, else `~/jevons/models`.
    pub fn models_folder(&self) -> PathBuf {
        self.models
            .folder
            .clone()
            .unwrap_or_else(|| user_dir().join("models"))
    }

    /// The provider called `name`: the settings' own, or the built-in `embedded`.
    pub fn provider(&self, name: &str) -> Option<Provider> {
        self.server().provider(name)
    }

    /// Where `capability` goes: its route, else `embedded`.
    pub fn route(&self, capability: Capability) -> Option<RouteTo> {
        self.server().route(capability)
    }

    /// What is wrong with the providers and routes, if anything.
    pub fn check_routes(&self) -> Result<(), String> {
        self.server().check_routes()
    }

    /// The key for the API the app exposes: `TYPESAFE_API_KEY`, then the settings.
    pub fn exposed_key(&self) -> Option<String> {
        self.server().exposed_key()
    }
}

/// These selections with each unset service filled from the first downloaded catalog entry
/// that serves it, in catalog order (DiffusionGemma first for language, then Parakeet for
/// speech). Explicit selections are kept.
pub fn with_downloaded(models: &Models, folder: &Path, catalog: &[CatalogEntry]) -> Models {
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
    let mut models = models.clone();
    if models.runtime_config.is_none() {
        models.generative = models.generative.or_else(|| pick(Service::Generative));
        models.decision = models.decision.or_else(|| pick(Service::Decision));
        models.speech = models.speech.or_else(|| pick(Service::Speech));
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_halves_make_the_settings_and_back() {
        let client: ClientConfig =
            toml::from_str(include_str!("../../../jevons-desktop.example.toml")).unwrap();
        let server: ServerConfig =
            toml::from_str(include_str!("../../../jevons-server.example.toml")).unwrap();
        let config = DesktopConfig::of(client.clone(), server.clone());
        assert_eq!((config.client(), config.server()), (client, server));
        assert_eq!(config.check_routes(), Ok(()));
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
            .branch_hotkeys
            .insert("ask".into(), "Ctrl+Alt+A".into());
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
        let catalog = jevons_desktop_core::catalog::builtin();
        for id in [
            "diffusiongemma-26b-a4b-q4_k_m",
            "nemotron-labs-diffusion-3b",
            "parakeet-tdt-0.6b-v3",
        ] {
            let dir = folder.join(id);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join(jevons_desktop_core::download::COMPLETE_MARKER),
                b"",
            )
            .unwrap();
        }
        let mine = Models {
            speech: Some(ModelRef {
                path: "/mine".into(),
                mmproj: None,
                catalog: None,
            }),
            ..Models::default()
        };
        let chosen = with_downloaded(&mine, &folder, &catalog);
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
    fn approvals_are_written_alone_and_saving_the_settings_keeps_them() {
        let dir = std::env::temp_dir().join(format!("jevons-approve-{}", std::process::id()));
        let file = dir.join("jevons-desktop.toml");
        let _ = std::fs::remove_dir_all(&dir);
        let mut panel = DesktopConfig::default();
        panel.dictation.language = Some("es".into());
        panel.save(&file).unwrap();
        let approved = DesktopConfig::approve(&file, "slack-post", "sha256:abc").unwrap();
        assert_eq!(approved.dictation.language.as_deref(), Some("es"));
        // A settings panel opened before the approval saves without dropping it.
        panel.dictation.language = Some("en".into());
        panel.save(&file).unwrap();
        let loaded = DesktopConfig::load(&file).unwrap();
        assert_eq!(loaded.dictation.language.as_deref(), Some("en"));
        assert_eq!(loaded.automation.approved["slack-post"], "sha256:abc");
        assert_eq!(
            loaded.automations_dir(&file),
            dir.join("automations"),
            "next to the settings by default"
        );
        std::fs::remove_dir_all(dir).unwrap();
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
