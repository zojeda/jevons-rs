//! `jevons-server.toml`: the desktop server's settings. Where inference comes from and which
//! provider serves each capability, the API it serves to other clients, the models the embedded
//! provider loads, the flows folder, the tools it runs and the API log. A missing file means
//! defaults.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub server: Server,
    /// Where inference may come from, by name. `embedded`, the models loaded in the app, is
    /// always there.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, Provider>,
    /// The provider and model of each capability.
    #[serde(skip_serializing_if = "Routes::is_default")]
    pub routes: Routes,
    pub models: Models,
    pub privacy: Logging,
    /// The flow tree's folder; defaults to `flows` next to this file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flows_dir: Option<PathBuf>,
    /// Built-in tools flow nodes may call, by name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, ToolConfig>,
    /// MCP servers whose tools flow nodes may call, as `server:tool`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp: BTreeMap<String, McpConfig>,
}

/// `[privacy]`: what the server writes down of what passes through it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Logging {
    /// Whether every decision and generation request and its response are written, whole, to
    /// the API log (they hold what the user said and the screen's text).
    pub log_api: bool,
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

impl ServerConfig {
    /// The flow tree's folder: the configured one, else `flows` next to the settings.
    pub fn flows_dir(&self, config_file: &Path) -> PathBuf {
        self.flows_dir
            .clone()
            .unwrap_or_else(|| config_file.parent().unwrap_or(Path::new(".")).join("flows"))
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
        let text = include_str!("../../../jevons-server.example.toml");
        let config: ServerConfig = toml::from_str(text).unwrap();
        assert!(!config.server.expose);
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.check_routes(), Ok(()));
    }

    fn routed(text: &str) -> ServerConfig {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn routes_left_out_go_to_the_embedded_provider() {
        let config = ServerConfig::default();
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
        let back: ServerConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
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
        assert!(toml::from_str::<ServerConfig>("[server]\nmode = \"remote\"\n").is_err());
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
    fn tools_and_mcp_servers_parse_and_check_their_placeholders() {
        let config: ServerConfig = toml::from_str(
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
        assert!(toml::from_str::<ServerConfig>(shell).is_err());
    }
}
