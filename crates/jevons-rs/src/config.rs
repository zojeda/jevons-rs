//! Server settings from flags, environment variables and an optional TOML file.
//!
//! Precedence: command-line flags, then environment variables, then the file, then defaults.
//! The file is `--config` (or `JEVONS_CONFIG`); otherwise `./jevons.toml` or
//! `~/.config/jevons/config.toml`, whichever exists first. Its keys are the long flag names
//! with underscores; relative paths and `~/` in it resolve against the file's directory and
//! the home directory.
use clap::{CommandFactory, FromArgMatches, Parser, parser::ValueSource};
use jevons_engine::Decoding;
use serde::Deserialize;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(about = "A System One and OpenAI-compatible API backed by diffusion language models")]
pub struct Settings {
    /// TOML settings file (see jevons.example.toml).
    #[arg(long, env = "JEVONS_CONFIG")]
    pub config: Option<PathBuf>,
    /// GGUF file or Hugging Face checkpoint directory.
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    pub model: Option<PathBuf>,
    /// Model architecture; detected from the model files by default.
    #[arg(long, env = "JEVONS_ARCH", default_value = "auto")]
    pub arch: String,
    /// Separate vision projector for DiffusionGemma; required for its image requests.
    #[arg(long, env = "DIFFUSION_MMPROJ")]
    pub mmproj: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: SocketAddr,
    /// Served model ID; defaults to the model's ID, such as gemmadiffusion-0.1 or
    /// nemotron-diffusion-3b.
    #[arg(long)]
    pub model_id: Option<String>,
    #[arg(long, env = "TYPESAFE_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// HIP device index.
    #[arg(long, default_value_t = 0)]
    pub main_gpu: usize,
    #[arg(long, default_value_t = 8192)]
    pub context_size: u32,
    #[arg(long, default_value_t = 512)]
    pub batch_size: u32,
    /// Recompute every prompt instead of reusing the longest cached token prefix.
    #[arg(long)]
    pub no_prompt_cache: bool,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 8)]
    pub queue_capacity: usize,
    /// How text (thoughts and OpenAI-compatible answers) is generated: diffusion,
    /// self-speculation or autoregressive. The last two need a masked model with causal
    /// predictions, such as Nemotron-Labs-Diffusion.
    #[arg(long, default_value = "diffusion")]
    pub decoding: Decoding,
}

/// The settings file. Every key is optional.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
struct File {
    model: Option<PathBuf>,
    arch: Option<String>,
    mmproj: Option<PathBuf>,
    bind: Option<SocketAddr>,
    model_id: Option<String>,
    api_key: Option<String>,
    main_gpu: Option<usize>,
    context_size: Option<u32>,
    batch_size: Option<u32>,
    prompt_cache: Option<bool>,
    seed: Option<u64>,
    queue_capacity: Option<usize>,
    decoding: Option<String>,
}

type Error = Box<dyn std::error::Error>;

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
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
    /// Parses the process arguments and merges the settings file.
    pub fn load() -> Result<Self, Error> {
        Self::load_from(std::env::args_os())
    }

    pub fn load_from<I, T>(args: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let matches = Self::command().try_get_matches_from(args)?;
        let mut settings = Self::from_arg_matches(&matches)?;
        let explicit = settings.config.is_some();
        if let Some(path) = settings.config.clone().or_else(default_file) {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("Cannot read settings file {}: {e}", path.display()))?;
            let file: File = toml::from_str(&text)
                .map_err(|e| format!("Invalid settings file {}: {e}", path.display()))?;
            let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            settings.merge(file, &base, |id| {
                matches!(
                    matches.value_source(id),
                    None | Some(ValueSource::DefaultValue)
                )
            })?;
            tracing::info!(file = %path.display(), explicit, "Loaded settings file");
            settings.config = Some(path);
        }
        Ok(settings)
    }

    /// Applies file values to every setting that `unset` reports as not given.
    fn merge(
        &mut self,
        file: File,
        base: &Path,
        unset: impl Fn(&str) -> bool,
    ) -> Result<(), Error> {
        macro_rules! take {
            ($field:ident) => {
                if let Some(value) = file.$field
                    && unset(stringify!($field))
                {
                    self.$field = value.into();
                }
            };
        }
        if let Some(model) = file.model
            && unset("model")
        {
            self.model = Some(resolve(model, base));
        }
        if let Some(mmproj) = file.mmproj
            && unset("mmproj")
        {
            self.mmproj = Some(resolve(mmproj, base));
        }
        take!(arch);
        take!(bind);
        take!(main_gpu);
        take!(context_size);
        take!(batch_size);
        take!(seed);
        take!(queue_capacity);
        if let Some(id) = file.model_id
            && unset("model_id")
        {
            self.model_id = Some(id);
        }
        if let Some(key) = file.api_key
            && unset("api_key")
        {
            self.api_key = Some(key);
        }
        if let Some(cache) = file.prompt_cache
            && unset("no_prompt_cache")
        {
            self.no_prompt_cache = !cache;
        }
        if let Some(decoding) = file.decoding
            && unset("decoding")
        {
            self.decoding = decoding.parse()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-config-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("jevons.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn the_file_fills_settings_that_flags_and_the_environment_do_not_give() {
        let file: File = toml::from_str(
            "model = \"models/3b\"\nbind = \"0.0.0.0:9000\"\ndecoding = \"self-speculation\"\n\
             prompt_cache = false\ncontext_size = 4096\nmodel_id = \"from-file\"\n",
        )
        .unwrap();
        let mut settings =
            Settings::try_parse_from(["jevons-rs", "--context-size", "2048"]).unwrap();
        let base = Path::new("/srv/jevons");
        // As if only --context-size were given.
        settings
            .merge(file, base, |id| id != "context_size")
            .unwrap();
        assert_eq!(settings.model, Some(base.join("models/3b")));
        assert_eq!(settings.bind, "0.0.0.0:9000".parse().unwrap());
        assert_eq!(settings.decoding, Decoding::SelfSpeculation);
        assert!(settings.no_prompt_cache);
        assert_eq!(settings.context_size, 2048, "flags win over the file");
        assert_eq!(settings.model_id.as_deref(), Some("from-file"));
    }

    #[test]
    fn a_settings_file_is_read_from_the_config_flag() {
        let path = write("load", "queue_capacity = 3\nseed = 7\n");
        let settings =
            Settings::load_from(["jevons-rs", "--config", path.to_str().unwrap()]).unwrap();
        assert_eq!((settings.queue_capacity, settings.seed), (3, 7));
        assert_eq!(settings.config, Some(path));
    }

    #[test]
    fn unknown_keys_and_bad_values_are_errors() {
        for (name, text) in [
            ("unknown", "modle = \"x\"\n"),
            ("decoding", "decoding = \"fast\"\n"),
            ("bind", "bind = \"nowhere\"\n"),
        ] {
            let path = write(name, text);
            assert!(
                Settings::load_from(["jevons-rs", "--config", path.to_str().unwrap()]).is_err(),
                "{text}"
            );
        }
        assert!(Settings::load_from(["jevons-rs", "--config", "/no/such/jevons.toml"]).is_err());
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
