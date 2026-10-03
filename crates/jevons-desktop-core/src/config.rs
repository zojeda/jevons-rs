//! `jevons-desktop.toml`: the providers and routes, models, dictation and privacy settings. The
//! Settings panel edits and saves it; a missing file means defaults.

use crate::context::Privacy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfig {
    pub server: Server,
    /// Where inference may come from, by name. `embedded`, the models loaded in this app, is
    /// always there.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, Provider>,
    /// The provider and model of each capability.
    #[serde(skip_serializing_if = "Routes::is_default")]
    pub routes: Routes,
    pub models: Models,
    pub dictation: Dictation,
    pub privacy: Privacy,
    /// The flow tree's folder; defaults to `flows` next to this file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flows_dir: Option<PathBuf>,
    /// Built-in tools flow nodes may call, by name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, ToolConfig>,
    /// MCP servers whose tools flow nodes may call, as `server:tool`.
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

/// What a built-in tool does.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Runs a program (never through a shell) with arguments and optional standard input.
    Command,
    /// Sends an HTTP request.
    Http,
    /// Opens an address or file with the system's default application.
    Open,
}

fn yes() -> bool {
    true
}

fn twenty() -> u64 {
    20
}

/// `[tools.<name>]`: a built-in tool. `{argument}` in its fields takes an argument's value, and
/// `${env:NAME}` an environment variable (for secrets, which never go in the flows folder).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolConfig {
    pub kind: ToolKind,
    /// What the tool does, for the model and for `TOOLS.md`.
    pub description: String,
    /// The tool's arguments, all text: name → what it is.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub arguments: BTreeMap<String, String>,
    /// `command`: the program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    /// `command`: its arguments, one per element.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// `command`: text for its standard input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdin: Option<String>,
    /// `command`: environment variables passed on (no others are).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<String>,
    /// `command`: the working folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// `http`: the method (POST by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// `http` and `open`: the address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// `http`: request headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// `http`: the body, usually JSON text with placeholders.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    /// Seconds before a call gives up.
    #[serde(default = "twenty")]
    pub timeout_s: u64,
    /// Ask in the bubble before each call (the default). Only the settings can turn it off.
    #[serde(default = "yes")]
    pub confirm: bool,
    /// Globs on the flow nodes that may call it, such as `["command/*"]`; empty allows all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
}

impl ToolConfig {
    /// What is wrong with it, if anything: missing fields, placeholders without an argument.
    pub fn check(&self) -> Result<(), String> {
        let need = |field: &Option<String>, name: &str| match field {
            Some(value) if !value.trim().is_empty() => Ok(()),
            _ => Err(format!("a {:?} tool needs `{name}`", self.kind)),
        };
        match self.kind {
            ToolKind::Command => need(&self.program, "program")?,
            ToolKind::Http | ToolKind::Open => need(&self.url, "url")?,
        }
        let mut texts: Vec<&str> = self.args.iter().map(String::as_str).collect();
        texts.extend(self.program.as_deref());
        texts.extend(self.stdin.as_deref());
        texts.extend(self.url.as_deref());
        texts.extend(self.body.as_deref());
        texts.extend(self.headers.values().map(String::as_str));
        for text in texts {
            for name in placeholders(text) {
                if !self.arguments.contains_key(&name) {
                    return Err(format!(
                        "{{{name}}} is not one of the tool's arguments ([arguments])"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The `{name}` placeholders in a tool field (`${env:…}` is not one).
pub fn placeholders(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        let is_env = start > 0 && rest[..start].ends_with('$');
        rest = &rest[start + 1..];
        let Some(end) = rest.find('}') else { break };
        if !is_env {
            out.push(rest[..end].to_string());
        }
        rest = &rest[end + 1..];
    }
    out
}

/// `[mcp.<name>]`: an MCP server started on demand, spoken to over its standard input and
/// output. Its tools are `name:tool` in flow files.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpConfig {
    /// The program and its arguments, such as `["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:/notes"]`.
    pub command: Vec<String>,
    /// Environment variables for it; values may use `${env:NAME}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Ask in the bubble before each call (the default).
    #[serde(default = "yes")]
    pub confirm: bool,
    /// Tools of this server that run without asking, such as read-only ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unconfirmed: Vec<String>,
    /// Globs on the flow nodes that may call its tools; empty allows all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
}

/// `[server]`: the API the app serves to other clients.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    /// Serve the API to other clients on `bind:port`. When off, the embedded API listens on an
    /// ephemeral loopback port with a key only this app knows.
    pub expose: bool,
    pub bind: IpAddr,
    pub port: u16,
    /// The key other clients use when exposed; `TYPESAFE_API_KEY` wins when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            expose: false,
            bind: IpAddr::from([127, 0, 0, 1]),
            port: 8080,
            api_key: None,
        }
    }
}

/// The provider every route uses unless it names another: the models loaded in this app.
pub const EMBEDDED: &str = "embedded";

/// What a provider is, which says where it is by default and what its decision model takes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    /// The models loaded in this app (`[models]`).
    Embedded,
    /// A jevons server elsewhere.
    Jevons,
    /// OpenRouter: System One (Jev) and generation.
    Openrouter,
    /// Any other server with OpenAI's API.
    OpenaiCompatible,
}

impl ProviderKind {
    /// The server root when the provider gives none.
    pub fn default_url(self) -> Option<&'static str> {
        match self {
            Self::Embedded | Self::OpenaiCompatible => None,
            Self::Jevons => Some("http://127.0.0.1:8080"),
            Self::Openrouter => Some("https://openrouter.ai/api"),
        }
    }

    /// The environment variable that holds the key when the provider gives none.
    pub fn default_key(self) -> Option<&'static str> {
        match self {
            Self::Embedded | Self::OpenaiCompatible => None,
            Self::Jevons => Some("TYPESAFE_API_KEY"),
            Self::Openrouter => Some("OPENROUTER_API_KEY"),
        }
    }

    /// Whether the provider says which model serves each capability (`GET /health`), so a
    /// route may leave its model out.
    pub fn names_its_models(self) -> bool {
        matches!(self, Self::Embedded | Self::Jevons)
    }
}

/// An extension to System One a decision provider may take.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Extension {
    Steps,
    Samples,
    Think,
}

/// `[providers.<name>]`: where a capability may be served from.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provider {
    pub kind: ProviderKind,
    /// The server root, without `/v1`. Every kind but `openai-compatible` has a default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The key; `${env:NAME}` takes it from the environment, so it stays out of the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// The System One extensions its decision model takes, in place of the kind's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<Extension>>,
    /// The most questions one decision request may ask, in place of the kind's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_questions: Option<usize>,
    /// The probability from which its decision model's choice counts as sure, in place of the
    /// kind's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_probability: Option<f64>,
}

impl Provider {
    pub fn of(kind: ProviderKind) -> Self {
        Self {
            kind,
            url: None,
            key: None,
            extensions: None,
            max_questions: None,
            min_probability: None,
        }
    }

    /// The server root, without a trailing slash; `None` for the embedded provider, whose
    /// address the app chooses.
    pub fn url(&self) -> Option<String> {
        self.url
            .as_deref()
            .or(self.kind.default_url())
            .map(|url| url.trim_end_matches('/').to_string())
            .filter(|url| !url.is_empty())
    }

    /// The key with `${env:NAME}` filled, else the kind's environment variable.
    pub fn key(&self) -> Option<String> {
        self.key
            .as_deref()
            .map(fill_env)
            .or_else(|| std::env::var(self.kind.default_key()?).ok())
            .filter(|key| !key.is_empty())
    }
}

/// `${env:NAME}` at the start of `text`: that environment variable (empty when unset) and what
/// follows it.
pub fn env_at(text: &str) -> Option<(String, &str)> {
    let after = text.strip_prefix("${env:")?;
    let end = after.find('}')?;
    let value = std::env::var(&after[..end]).unwrap_or_default();
    Some((value, &after[end + 1..]))
}

/// `text` with each `${env:NAME}` replaced by that environment variable.
pub fn fill_env(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("${env:") {
        let Some((value, after)) = env_at(&rest[start..]) else {
            break;
        };
        out.push_str(&rest[..start]);
        out.push_str(&value);
        rest = after;
    }
    out.push_str(rest);
    out
}

/// What the app asks of a model.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Capability {
    /// Uploaded audio to text.
    Speech,
    /// Streamed audio to text.
    Realtime,
    /// System One.
    Decision,
    /// Responses and chat.
    Generation,
}

impl Capability {
    pub const ALL: [Self; 4] = [
        Self::Speech,
        Self::Realtime,
        Self::Decision,
        Self::Generation,
    ];

    /// Its key in `[routes]`.
    pub fn key(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::Realtime => "realtime",
            Self::Decision => "decision",
            Self::Generation => "generation",
        }
    }
}

/// One of `[routes]`: the provider of a capability, and the model to ask it for.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteTo {
    pub provider: String,
    /// The model's name at the provider. An embedded or jevons provider names its own when
    /// this is left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// `[routes]`: each capability's provider. One left out goes to `embedded`; `realtime` left out
/// follows `speech`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Routes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech: Option<RouteTo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime: Option<RouteTo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<RouteTo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<RouteTo>,
}

impl Routes {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn get(&self, capability: Capability) -> Option<&RouteTo> {
        match capability {
            Capability::Speech => self.speech.as_ref(),
            Capability::Realtime => self.realtime.as_ref(),
            Capability::Decision => self.decision.as_ref(),
            Capability::Generation => self.generation.as_ref(),
        }
    }

    pub fn set(&mut self, capability: Capability, route: Option<RouteTo>) {
        *match capability {
            Capability::Speech => &mut self.speech,
            Capability::Realtime => &mut self.realtime,
            Capability::Decision => &mut self.decision,
            Capability::Generation => &mut self.generation,
        } = route;
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

    /// Writes `file`, creating its folder. The approvals are not the settings panel's to change:
    /// they are kept as the file has them.
    pub fn save(&self, file: &Path) -> Result<(), ConfigError> {
        let mut saved = self.clone();
        if let Ok(current) = Self::load(file) {
            saved.automation.approved = current.automation.approved;
        }
        saved.write(file)
    }

    /// Pins `version` as the approved one of automation `name`, changing nothing else in the
    /// file; returns the settings as saved.
    pub fn approve(file: &Path, name: &str, version: &str) -> Result<Self, ConfigError> {
        let mut config = Self::load(file)?;
        config
            .automation
            .approved
            .insert(name.to_string(), version.to_string());
        config.write(file)?;
        Ok(config)
    }

    fn write(&self, file: &Path) -> Result<(), ConfigError> {
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

    pub fn flows_dir(&self, config_file: &Path) -> PathBuf {
        self.flows_dir
            .clone()
            .unwrap_or_else(|| config_file.parent().unwrap_or(Path::new(".")).join("flows"))
    }

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

    /// The models folder: the configured one, else `~/jevons/models`.
    pub fn models_folder(&self) -> PathBuf {
        self.models
            .folder
            .clone()
            .unwrap_or_else(|| user_dir().join("models"))
    }

    /// The provider called `name`: the settings' own, or the built-in `embedded`.
    pub fn provider(&self, name: &str) -> Option<Provider> {
        self.providers
            .get(name)
            .cloned()
            .or_else(|| (name == EMBEDDED).then(|| Provider::of(ProviderKind::Embedded)))
    }

    /// Where `capability` goes: its route, else `embedded`. Realtime left out follows speech,
    /// when speech is on a provider that may stream (embedded or jevons).
    pub fn route(&self, capability: Capability) -> Option<RouteTo> {
        if let Some(route) = self.routes.get(capability) {
            return Some(route.clone());
        }
        if capability == Capability::Realtime {
            let speech = self.route(Capability::Speech)?;
            let kind = self.provider(&speech.provider)?.kind;
            return kind.names_its_models().then_some(speech);
        }
        Some(RouteTo {
            provider: EMBEDDED.into(),
            model: None,
        })
    }

    /// What is wrong with the providers and routes, if anything: a route to a provider that
    /// is not there, a provider with no address, a route with no model where the provider
    /// cannot name its own, an override out of range, or one model name on two providers
    /// (other clients' requests are forwarded by model name).
    pub fn check_routes(&self) -> Result<(), String> {
        for (name, provider) in &self.providers {
            if provider.kind != ProviderKind::Embedded && provider.url().is_none() {
                return Err(format!("[providers.{name}] needs a `url`"));
            }
            if provider.max_questions == Some(0) {
                return Err(format!("[providers.{name}] max_questions is at least 1"));
            }
            if provider
                .min_probability
                .is_some_and(|p| !(0.0..=1.0).contains(&p))
            {
                return Err(format!(
                    "[providers.{name}] min_probability is a probability from 0 to 1"
                ));
            }
        }
        let mut served: BTreeMap<String, String> = BTreeMap::new();
        for capability in Capability::ALL {
            let Some(route) = self.route(capability) else {
                continue;
            };
            let key = capability.key();
            let Some(provider) = self.provider(&route.provider) else {
                return Err(format!(
                    "[routes] {key} names the provider {:?}, which [providers] does not have",
                    route.provider
                ));
            };
            match &route.model {
                Some(model) => {
                    if let Some(other) = served.insert(model.clone(), route.provider.clone())
                        && other != route.provider
                    {
                        return Err(format!(
                            "[routes] the model {model:?} is asked of both {other} and {}: a \
                             model name goes to one provider",
                            route.provider
                        ));
                    }
                }
                None if provider.kind.names_its_models() => {}
                None => {
                    return Err(format!(
                        "[routes] {key} needs a `model`: {} does not say which it serves",
                        route.provider
                    ));
                }
            }
        }
        Ok(())
    }

    /// The key for the API the app exposes: `TYPESAFE_API_KEY`, then the settings.
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
        assert_eq!(config.check_routes(), Ok(()));
    }

    fn routed(text: &str) -> DesktopConfig {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn routes_left_out_go_to_the_embedded_provider() {
        let config = DesktopConfig::default();
        assert_eq!(config.check_routes(), Ok(()));
        for capability in Capability::ALL {
            let route = config.route(capability).unwrap();
            assert_eq!((route.provider.as_str(), route.model), (EMBEDDED, None));
        }
        assert_eq!(
            config.provider(EMBEDDED).unwrap().kind,
            ProviderKind::Embedded
        );
        // Nothing of it is written to a file that never set it.
        let text = toml::to_string(&config).unwrap();
        assert!(
            !text.contains("providers") && !text.contains("routes"),
            "{text}"
        );
    }

    #[test]
    fn each_capability_goes_to_the_provider_its_route_names() {
        let config = routed(
            r#"
[providers.box]
kind = "jevons"
url = "http://box:8080/"

[providers.openrouter]
kind = "openrouter"
max_questions = 4
min_probability = 0.6
extensions = ["think"]

[routes]
speech = { provider = "box" }
decision = { provider = "openrouter", model = "typesafe/jev-1.13" }
"#,
        );
        assert_eq!(config.check_routes(), Ok(()));
        let decision = config.route(Capability::Decision).unwrap();
        assert_eq!(decision.provider, "openrouter");
        assert_eq!(decision.model.as_deref(), Some("typesafe/jev-1.13"));
        let openrouter = config.provider("openrouter").unwrap();
        assert_eq!(
            openrouter.url().as_deref(),
            Some("https://openrouter.ai/api")
        );
        assert_eq!(openrouter.extensions, Some(vec![Extension::Think]));
        // Realtime follows speech to a provider that streams; generation stays embedded.
        assert_eq!(config.route(Capability::Realtime).unwrap().provider, "box");
        assert_eq!(
            config.provider("box").unwrap().url().as_deref(),
            Some("http://box:8080")
        );
        assert_eq!(
            config.route(Capability::Generation).unwrap().provider,
            EMBEDDED
        );
        // Speech on a provider that only takes uploads leaves Realtime off.
        let uploads = routed(
            "[providers.cloud]\nkind = \"openai-compatible\"\nurl = \"https://stt.example\"\n\
             [routes]\nspeech = { provider = \"cloud\", model = \"whisper-1\" }\n",
        );
        assert_eq!(uploads.check_routes(), Ok(()));
        assert_eq!(uploads.route(Capability::Realtime), None);
        // Saved, it loads back the same.
        let back: DesktopConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn providers_and_routes_that_cannot_work_are_errors() {
        let error = |text: &str| routed(text).check_routes().unwrap_err();
        assert_eq!(
            error("[routes]\ndecision = { provider = \"nowhere\" }\n"),
            "[routes] decision names the provider \"nowhere\", which [providers] does not have"
        );
        assert_eq!(
            error("[providers.other]\nkind = \"openai-compatible\"\n"),
            "[providers.other] needs a `url`"
        );
        assert_eq!(
            error(
                "[providers.openrouter]\nkind = \"openrouter\"\n\
                 [routes]\ngeneration = { provider = \"openrouter\" }\n"
            ),
            "[routes] generation needs a `model`: openrouter does not say which it serves"
        );
        assert_eq!(
            error("[providers.openrouter]\nkind = \"openrouter\"\nmax_questions = 0\n"),
            "[providers.openrouter] max_questions is at least 1"
        );
        assert_eq!(
            error("[providers.openrouter]\nkind = \"openrouter\"\nmin_probability = 1.5\n"),
            "[providers.openrouter] min_probability is a probability from 0 to 1"
        );
        assert!(
            error(
                "[providers.a]\nkind = \"openrouter\"\n[providers.b]\nkind = \"jevons\"\n\
                 [routes]\ndecision = { provider = \"a\", model = \"m\" }\n\
                 generation = { provider = \"b\", model = \"m\" }\n"
            )
            .starts_with("[routes] the model \"m\" is asked of both a and b")
        );
        // The mode of earlier settings files is gone.
        assert!(toml::from_str::<DesktopConfig>("[server]\nmode = \"remote\"\n").is_err());
    }

    #[test]
    fn a_provider_s_key_comes_from_the_file_or_the_environment() {
        let path = std::env::var("PATH").unwrap();
        let mut provider = Provider::of(ProviderKind::OpenaiCompatible);
        assert_eq!(provider.key(), None);
        provider.key = Some("${env:PATH}".into());
        assert_eq!(provider.key(), Some(path.clone()));
        assert_eq!(fill_env("a ${env:PATH} b"), format!("a {path} b"));
        assert_eq!(fill_env("${env:JEVONS_NO_SUCH_VARIABLE}"), "");
        assert_eq!(fill_env("plain ${env:open"), "plain ${env:open");
        provider.key = Some("sk-literal".into());
        assert_eq!(provider.key().as_deref(), Some("sk-literal"));
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
    fn a_live_hotkey_turned_off_stays_off_once_saved() {
        let mut config = DesktopConfig::default();
        config.dictation.live_hotkey = None;
        let text = toml::to_string(&config).unwrap();
        assert!(text.contains("live_hotkey = \"\""), "{text}");
        let back: DesktopConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.dictation.live_hotkey, None);
        // Left out, it is the default.
        let default: DesktopConfig = toml::from_str("").unwrap();
        assert_eq!(default.dictation.live_hotkey.as_deref(), Some("F9"));
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
    fn tools_and_mcp_servers_parse_and_check_their_placeholders() {
        let config: DesktopConfig = toml::from_str(
            r#"
[tools.search]
kind = "open"
description = "Searches the web"
url = "https://duckduckgo.com/?q={query}"
arguments = { query = "What to search for" }
confirm = false

[tools.note]
kind = "command"
description = "Saves a note"
program = "notes.exe"
args = ["--title", "{title}", "--folder", "{folder}"]

[mcp.fs]
command = ["npx", "-y", "server-filesystem", "C:/notes"]
unconfirmed = ["read_file"]
"#,
        )
        .unwrap();
        assert!(config.tools["search"].check().is_ok());
        assert!(!config.tools["search"].confirm);
        let error = config.tools["note"].check().unwrap_err();
        assert!(error.contains("{title}"), "{error}");
        assert!(config.mcp["fs"].confirm, "servers ask by default");
        assert_eq!(placeholders("Bearer ${env:TOKEN} {id}"), ["id"]);
        let shell = "[tools.x]\nkind = \"shell\"\ndescription = \"\"";
        assert!(toml::from_str::<DesktopConfig>(shell).is_err());
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
