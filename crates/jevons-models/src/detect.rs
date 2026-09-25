//! Identifies a model's architecture from its files.
use jevons_core::{Error, Result};
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// DiffusionGemma GGUF (`general.architecture = diffusion-gemma`).
    Gemma4Diffusion,
    /// Nemotron-Labs-Diffusion Hugging Face checkpoint directory (text or VLM).
    NemotronDiffusion,
}

impl Architecture {
    pub const ALL: [Self; 2] = [Self::Gemma4Diffusion, Self::NemotronDiffusion];

    /// The `--arch` value.
    pub fn id(self) -> &'static str {
        match self {
            Self::Gemma4Diffusion => "gemma4-diffusion",
            Self::NemotronDiffusion => "nemotron-diffusion",
        }
    }

    /// Whether image input needs a separate vision projector file (`--mmproj`).
    pub fn uses_separate_projector(self) -> bool {
        matches!(self, Self::Gemma4Diffusion)
    }

    /// Routing alias that names the newest local model of this architecture.
    pub fn latest_alias(self) -> &'static str {
        match self {
            Self::Gemma4Diffusion => "gemmadiffusion-latest",
            Self::NemotronDiffusion => "nemotron-diffusion-latest",
        }
    }

    /// The model ID served by default for this architecture when the files do not name a size.
    pub fn default_model_id(self) -> &'static str {
        match self {
            Self::Gemma4Diffusion => "gemmadiffusion-0.1",
            Self::NemotronDiffusion => "nemotron-diffusion-8b",
        }
    }
}

/// The model ID served by default for the model at `path`: the architecture's ID, sized for
/// Nemotron-Labs-Diffusion checkpoints (`nemotron-diffusion-3b`, `-8b` or `-14b`).
pub fn default_model_id(path: &Path, architecture: Architecture) -> String {
    let size = match architecture {
        Architecture::NemotronDiffusion => checkpoint_dir(path)
            .and_then(|dir| std::fs::read_to_string(dir.join("config.json")).ok())
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|config| {
                let dims = (
                    config["num_hidden_layers"].as_u64()?,
                    config["hidden_size"].as_u64()?,
                );
                match dims {
                    (26, 3072) => Some("3b"),
                    (34, 4096) => Some("8b"),
                    (40, 5120) => Some("14b"),
                    _ => None,
                }
            }),
        Architecture::Gemma4Diffusion => None,
    };
    match size {
        Some(size) => format!("nemotron-diffusion-{size}"),
        None => architecture.default_model_id().into(),
    }
}

impl std::str::FromStr for Architecture {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|a| a.id() == name)
            .ok_or_else(|| {
                let known: Vec<_> = Self::ALL.iter().map(|a| a.id()).collect();
                Error::InvalidInput(format!(
                    "Unknown architecture {name:?}; expected auto or one of {}",
                    known.join(", ")
                ))
            })
    }
}

/// The directory holding a Hugging Face checkpoint named by `path`: the directory itself, or
/// the parent of its `config.json` or a `.safetensors` file inside it.
pub fn checkpoint_dir(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    let name = path.file_name()?.to_str()?;
    (name == "config.json"
        || name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json"))
    .then(|| path.parent().map(Path::to_path_buf))
    .flatten()
}

pub fn detect(path: &Path) -> Result<Architecture> {
    if let Some(dir) = checkpoint_dir(path) {
        return detect_checkpoint(&dir);
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map_err(|_| Error::ModelLoad)?;
    if &magic != b"GGUF" {
        return Err(Error::UnsupportedModel(format!(
            "{} is neither a GGUF file nor a Hugging Face checkpoint directory",
            path.display()
        )));
    }
    let gguf = jevons_formats::gguf::Gguf::open(path).map_err(|_| Error::ModelLoad)?;
    match gguf
        .get("general.architecture")
        .ok()
        .and_then(|value| value.as_str())
    {
        Some("diffusion-gemma") => Ok(Architecture::Gemma4Diffusion),
        other => Err(Error::UnsupportedModel(format!(
            "GGUF architecture {} is not supported",
            other.unwrap_or("(missing)")
        ))),
    }
}

fn detect_checkpoint(dir: &Path) -> Result<Architecture> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|_| Error::ModelLoad)?;
    let config: serde_json::Value = serde_json::from_str(&text).map_err(|_| Error::ModelLoad)?;
    let model_type = config["model_type"].as_str().unwrap_or_default();
    let architectures: Vec<&str> = config["architectures"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if matches!(
        model_type,
        "nemotron_labs_diffusion" | "nemotron_labs_diffusion_vlm"
    ) || architectures.iter().any(|a| {
        matches!(
            *a,
            "NemotronLabsDiffusionModel" | "NemotronLabsDiffusionVLMModel"
        )
    }) {
        return Ok(Architecture::NemotronDiffusion);
    }
    if crate::speech::is_speech_checkpoint(&config) {
        return Err(Error::UnsupportedModel(format!(
            "{} is a speech-to-text model; use it for services.speech",
            dir.display()
        )));
    }
    Err(Error::UnsupportedModel(format!(
        "checkpoint model_type {model_type:?} is not supported"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-detect-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn architecture_names_round_trip_and_unknown_names_are_rejected() {
        for arch in Architecture::ALL {
            assert_eq!(arch.id().parse::<Architecture>().unwrap(), arch);
        }
        assert!("gemma".parse::<Architecture>().is_err());
    }

    #[test]
    fn checkpoint_directories_are_detected_from_config_json() {
        let dir = temp_dir("nemotron");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"nemotron_labs_diffusion_vlm","architectures":["NemotronLabsDiffusionVLMModel"]}"#,
        )
        .unwrap();
        assert_eq!(detect(&dir).unwrap(), Architecture::NemotronDiffusion);
        assert_eq!(
            detect(&dir.join("config.json")).unwrap(),
            Architecture::NemotronDiffusion
        );
        assert_eq!(
            default_model_id(&dir, Architecture::NemotronDiffusion),
            "nemotron-diffusion-8b"
        );
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"nemotron_labs_diffusion","architectures":["NemotronLabsDiffusionModel"],"num_hidden_layers":26,"hidden_size":3072}"#,
        )
        .unwrap();
        assert_eq!(detect(&dir).unwrap(), Architecture::NemotronDiffusion);
        assert_eq!(
            default_model_id(&dir, Architecture::NemotronDiffusion),
            "nemotron-diffusion-3b"
        );
        std::fs::write(dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        assert!(matches!(detect(&dir), Err(Error::UnsupportedModel(_))));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn files_without_gguf_magic_are_unsupported() {
        let dir = temp_dir("magic");
        let file = dir.join("model.bin");
        std::fs::write(&file, b"not a model").unwrap();
        assert!(matches!(detect(&file), Err(Error::UnsupportedModel(_))));
        assert!(matches!(
            detect(&dir.join("missing.gguf")),
            Err(Error::ModelLoad)
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
