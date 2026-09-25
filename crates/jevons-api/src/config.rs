//! Server settings: a TOML file of loaded models and the services that use them.
//!
//! ```toml
//! [server]
//! bind = "127.0.0.1:8080"
//!
//! [models.gemma]                       # a model, loaded once
//! path = "~/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
//!
//! [services.generative]                # OpenAI chat, completions and responses
//! model = "gemma"
//! [services.decision]                  # System One, on the same loaded engine
//! model = "gemma"
//! ```
//!
//! The file is `--config` (or `JEVONS_CONFIG`); otherwise `./jevons.toml` or
//! `~/.config/jevons/config.toml`, whichever exists first. `--bind` and the API key
//! (`TYPESAFE_API_KEY`) override the file. Relative paths and `~/` resolve against the file's
//! directory and the home directory. See `jevons.example.toml` for every key.
use clap::{CommandFactory, FromArgMatches, Parser};
use jevons_diffusion::Decoding;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::{Path, PathBuf},
};

/// The release build (`JEVONS_BUILD`, set by CI) or the crate version.
const VERSION: &str = match option_env!("JEVONS_BUILD") {
    Some(build) => build,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(
    about = "A personal inference runtime: OpenAI-compatible chat, transcription and Realtime, and System One APIs on Burn and CubeCL",
    version = VERSION
)]
struct Args {
    /// TOML settings file of models and services (see jevons.example.toml).
    #[arg(long, env = "JEVONS_CONFIG")]
    config: Option<PathBuf>,
    /// Listen address; overrides `server.bind`.
    #[arg(long)]
    bind: Option<SocketAddr>,
    /// Bearer key for every endpoint except /health; overrides `server.api_key`.
    #[arg(long, env = "TYPESAFE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
}

type Error = Box<dyn std::error::Error>;

/// Everything the server runs: where it listens, the models it loads, and the services that
/// use them.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The file the settings came from.
    pub file: PathBuf,
    pub server: Server,
    /// Models by name, each loaded once.
    pub models: BTreeMap<String, Model>,
    pub services: Services,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Server {
    #[serde(default = "default_bind")]
    pub bind: SocketAddr,
    /// Prefer `TYPESAFE_API_KEY`; never commit a key.
    #[serde(default)]
    pub api_key: Option<String>,
}

fn default_bind() -> SocketAddr {
    "127.0.0.1:8080".parse().expect("a valid default address")
}

/// A model to load. Options other than `path`, `id`, `main_gpu` and `queue_capacity` apply to
/// diffusion language models only; they are unset (`None`) unless the file names them, so a
/// speech model that sets one is an error.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// GGUF file (DiffusionGemma) or Hugging Face checkpoint directory.
    pub path: PathBuf,
    /// Served model ID; defaults to the model's own, such as `gemmadiffusion-0.1`,
    /// `nemotron-diffusion-3b` or `parakeet-tdt-0.6b-v3`.
    #[serde(default)]
    pub id: Option<String>,
    /// HIP device index.
    #[serde(default)]
    pub main_gpu: usize,
    /// Requests waiting for this model at once.
    #[serde(default = "default_queue")]
    pub queue_capacity: usize,
    /// Architecture (`gemma4-diffusion` or `nemotron-diffusion`); detected from the files by
    /// default.
    #[serde(default)]
    pub arch: Option<String>,
    /// Separate vision projector for DiffusionGemma's image input.
    #[serde(default)]
    pub mmproj: Option<PathBuf>,
    #[serde(default)]
    pub context_size: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    /// Reuse the longest cached token prefix between requests (default true).
    #[serde(default)]
    pub prompt_cache: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// `diffusion`, `self-speculation` or `autoregressive`; the last two need a masked model
    /// with causal predictions, such as Nemotron-Labs-Diffusion.
    #[serde(default)]
    pub decoding: Option<String>,
}

fn default_queue() -> usize {
    8
}

impl Model {
    pub fn context_size(&self) -> u32 {
        self.context_size.unwrap_or(8192)
    }

    pub fn batch_size(&self) -> u32 {
        self.batch_size.unwrap_or(512)
    }

    pub fn prompt_cache(&self) -> bool {
        self.prompt_cache.unwrap_or(true)
    }

    pub fn seed(&self) -> u64 {
        self.seed.unwrap_or(42)
    }

    pub fn decoding(&self) -> Result<Decoding, Error> {
        Ok(match &self.decoding {
            Some(name) => name.parse()?,
            None => Decoding::default(),
        })
    }

    /// The diffusion-only options this model sets, which a speech model must not.
    pub fn diffusion_options(&self) -> Vec<&'static str> {
        [
            ("arch", self.arch.is_some()),
            ("mmproj", self.mmproj.is_some()),
            ("context_size", self.context_size.is_some()),
            ("batch_size", self.batch_size.is_some()),
            ("prompt_cache", self.prompt_cache.is_some()),
            ("seed", self.seed.is_some()),
            ("decoding", self.decoding.is_some()),
        ]
        .into_iter()
        .filter_map(|(key, set)| set.then_some(key))
        .collect()
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Services {
    /// OpenAI Chat Completions, Completions and Responses.
    #[serde(default)]
    pub generative: Option<ServiceModel>,
    /// The System One API.
    #[serde(default)]
    pub decision: Option<ServiceModel>,
    /// OpenAI audio transcriptions and Realtime transcription sessions.
    #[serde(default)]
    pub speech: Option<Speech>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServiceModel {
    /// The `[models.<name>]` this service runs on.
    pub model: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Speech {
    pub model: String,
    /// Longest audio upload or Realtime input buffer, in seconds.
    #[serde(default = "default_max_audio")]
    pub max_audio_seconds: f64,
    /// Whether `/v1/realtime` is served.
    #[serde(default = "default_true")]
    pub realtime: bool,
}

fn default_max_audio() -> f64 {
    3600.0
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    server: Option<Server>,
    #[serde(default)]
    models: BTreeMap<String, Model>,
    #[serde(default)]
    services: Services,
}

/// `HOME`, or `USERPROFILE` on Windows.
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// `~/...` from the home directory; other relative paths from `base`.
fn resolve(path: PathBuf, base: &Path) -> PathBuf {
    match (path.strip_prefix("~"), home()) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ if path.is_relative() => base.join(path),
        _ => path,
    }
}

fn default_file() -> Option<PathBuf> {
    let local = PathBuf::from("jevons.toml");
    if local.is_file() {
        return Some(local);
    }
    let user = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| home().map(|h| h.join(".config")))?
        .join("jevons/config.toml");
    user.is_file().then_some(user)
}

impl Settings {
    /// Parses the process arguments and reads the settings file. `--help`, `--version` and
    /// invalid flags print their message and exit, as usual for clap.
    pub fn load() -> Result<Self, Error> {
        Self::from_args(Args::from_arg_matches(&Args::command().get_matches())?)
    }

    pub fn load_from<I, T>(args: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        Self::from_args(Args::try_parse_from(args)?)
    }

    fn from_args(args: Args) -> Result<Self, Error> {
        let path = args.config.or_else(default_file).ok_or(
            "No settings file: pass --config PATH (or JEVONS_CONFIG), or create ./jevons.toml \
             or ~/.config/jevons/config.toml from jevons.example.toml",
        )?;
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read settings file {}: {e}", path.display()))?;
        let mut settings = Self::parse(&text, &path)?;
        if let Some(bind) = args.bind {
            settings.server.bind = bind;
        }
        if let Some(key) = args.api_key {
            settings.server.api_key = Some(key);
        }
        if settings
            .server
            .api_key
            .as_ref()
            .is_some_and(|k| k.is_empty())
        {
            return Err("The configured API key must be nonempty".into());
        }
        tracing::info!(file = %path.display(), "Loaded settings file");
        Ok(settings)
    }

    /// Parses and validates a settings file read from `path`.
    pub fn parse(text: &str, path: &Path) -> Result<Self, Error> {
        let invalid = |e: String| -> Error {
            format!("Invalid settings file {}: {e}", path.display()).into()
        };
        let file: File = toml::from_str(text).map_err(|e| invalid(e.to_string()))?;
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut models = file.models;
        for model in models.values_mut() {
            model.path = resolve(std::mem::take(&mut model.path), &base);
            model.mmproj = model.mmproj.take().map(|p| resolve(p, &base));
        }
        let settings = Self {
            file: path.to_path_buf(),
            server: file.server.unwrap_or(Server {
                bind: default_bind(),
                api_key: None,
            }),
            models,
            services: file.services,
        };
        settings.validate().map_err(invalid)?;
        Ok(settings)
    }

    fn validate(&self) -> Result<(), String> {
        let s = &self.services;
        let used: Vec<(&str, &str)> = [
            (
                "generative",
                s.generative.as_ref().map(|g| g.model.as_str()),
            ),
            ("decision", s.decision.as_ref().map(|d| d.model.as_str())),
            ("speech", s.speech.as_ref().map(|d| d.model.as_str())),
        ]
        .into_iter()
        .filter_map(|(service, model)| Some((service, model?)))
        .collect();
        if used.is_empty() {
            return Err(
                "enable at least one of services.generative, services.decision and \
                        services.speech"
                    .into(),
            );
        }
        for (service, model) in &used {
            if !self.models.contains_key(*model) {
                return Err(format!(
                    "services.{service} uses model {model:?}, which has no [models.{model}]"
                ));
            }
        }
        if let Some(speech) = &s.speech {
            if used
                .iter()
                .any(|(service, model)| *service != "speech" && *model == speech.model)
            {
                return Err(format!(
                    "model {:?} serves speech and a language service; a model is one or the other",
                    speech.model
                ));
            }
            if !(speech.max_audio_seconds.is_finite() && speech.max_audio_seconds > 0.0) {
                return Err("services.speech.max_audio_seconds must be positive".into());
            }
        }
        for (name, model) in &self.models {
            if !used.iter().any(|(_, m)| m == name) {
                return Err(format!("model {name:?} is not used by any service"));
            }
            if model.queue_capacity == 0 {
                return Err(format!("models.{name}.queue_capacity must be positive"));
            }
            if model.id.as_ref().is_some_and(|id| id.trim().is_empty()) {
                return Err(format!("models.{name}.id must be nonempty"));
            }
            model
                .decoding()
                .map_err(|e| format!("models.{name}.decoding: {e}"))?;
        }
        Ok(())
    }

    /// The names of the models that serve diffusion (Generative or Decision) services.
    pub fn diffusion_models(&self) -> Vec<&str> {
        let mut names: Vec<&str> = [&self.services.generative, &self.services.decision]
            .into_iter()
            .flatten()
            .map(|s| s.model.as_str())
            .collect();
        names.dedup();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Settings, Error> {
        Settings::parse(text, Path::new("/srv/jevons/jevons.toml"))
    }

    const SHARED: &str = r#"
[server]
bind = "0.0.0.0:9000"

[models.gemma]
path = "models/gemma.gguf"
mmproj = "~/models/mmproj.gguf"
context_size = 4096
decoding = "diffusion"

[models.parakeet]
path = "/models/parakeet"
id = "asr"

[services.generative]
model = "gemma"
[services.decision]
model = "gemma"
[services.speech]
model = "parakeet"
max_audio_seconds = 600.0
realtime = false
"#;

    #[test]
    fn services_share_a_model_declared_once() {
        let settings = parse(SHARED).unwrap();
        assert_eq!(settings.server.bind, "0.0.0.0:9000".parse().unwrap());
        let gemma = &settings.models["gemma"];
        assert_eq!(gemma.path, PathBuf::from("/srv/jevons/models/gemma.gguf"));
        if let Some(home) = home() {
            assert_eq!(gemma.mmproj, Some(home.join("models/mmproj.gguf")));
        }
        assert_eq!(
            (gemma.context_size(), gemma.batch_size(), gemma.seed()),
            (4096, 512, 42)
        );
        assert!(gemma.prompt_cache());
        assert_eq!(settings.diffusion_models(), vec!["gemma"]);
        let speech = settings.services.speech.as_ref().unwrap();
        assert_eq!((speech.max_audio_seconds, speech.realtime), (600.0, false));
        assert_eq!(settings.models["parakeet"].id.as_deref(), Some("asr"));
        assert!(settings.models["parakeet"].diffusion_options().is_empty());
        assert_eq!(
            gemma.diffusion_options(),
            vec!["mmproj", "context_size", "decoding"]
        );
    }

    #[test]
    fn generative_and_decision_may_use_different_models() {
        let settings = parse(
            r#"
[models.gemma]
path = "/m/gemma.gguf"
[models.nemotron]
path = "/m/nemotron"
decoding = "self-speculation"
[services.decision]
model = "gemma"
[services.generative]
model = "nemotron"
"#,
        )
        .unwrap();
        assert_eq!(settings.diffusion_models(), vec!["nemotron", "gemma"]);
        assert_eq!(settings.server.bind, default_bind());
        assert_eq!(
            settings.models["nemotron"].decoding().unwrap(),
            Decoding::SelfSpeculation
        );
    }

    #[test]
    fn inconsistent_settings_are_errors() {
        for (text, expected) in [
            ("[models.a]\npath = \"x\"\n", "at least one"),
            (
                "[services.generative]\nmodel = \"a\"\n",
                "has no [models.a]",
            ),
            (
                "[models.a]\npath = \"x\"\n[models.b]\npath = \"y\"\n[services.decision]\nmodel = \"a\"\n",
                "not used",
            ),
            (
                "[models.a]\npath = \"x\"\n[services.decision]\nmodel = \"a\"\n[services.speech]\nmodel = \"a\"\n",
                "one or the other",
            ),
            (
                "[models.a]\npath = \"x\"\ndecoding = \"fast\"\n[services.decision]\nmodel = \"a\"\n",
                "decoding",
            ),
            ("[models.a]\npath = \"x\"\nmodle = 1\n", "unknown field"),
            ("model = \"x\"\n", "unknown field"),
            ("[server]\nbind = \"nowhere\"\n", "bind"),
        ] {
            let error = parse(text).unwrap_err().to_string();
            assert!(error.contains(expected), "{text:?}: {error}");
        }
    }

    fn write(name: &str, text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-config-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("jevons.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn flags_override_the_file_and_a_missing_file_is_an_error() {
        let path = write("load", SHARED);
        let settings = Settings::load_from([
            "jevons-rs",
            "--config",
            path.to_str().unwrap(),
            "--bind",
            "127.0.0.1:7000",
        ])
        .unwrap();
        assert_eq!(settings.server.bind, "127.0.0.1:7000".parse().unwrap());
        assert_eq!(settings.file, path);
        assert!(Settings::load_from(["jevons-rs", "--config", "/no/such/jevons.toml"]).is_err());
        // The flat model flags are gone.
        assert!(Settings::load_from(["jevons-rs", "-m", "model.gguf"]).is_err());
    }

    #[test]
    fn home_relative_paths_expand() {
        let base = Path::new("/etc/jevons");
        if let Some(home) = home() {
            assert_eq!(resolve("~/models/x".into(), base), home.join("models/x"));
        }
        assert_eq!(resolve("x".into(), base), base.join("x"));
        assert_eq!(resolve("/x".into(), base), PathBuf::from("/x"));
    }
}
