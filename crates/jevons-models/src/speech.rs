//! Speech-to-text architecture detection and loading.
use crate::detect::checkpoint_dir;
use jevons_core::{Error, Result, SpeechConfig, SpeechModel};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechArchitecture {
    /// NVIDIA Parakeet TDT Hugging Face checkpoint directory (`ParakeetForTDT`).
    ParakeetTdt,
}

impl SpeechArchitecture {
    pub fn id(self) -> &'static str {
        match self {
            Self::ParakeetTdt => "parakeet-tdt",
        }
    }

    /// Routing alias that names the newest local model of this architecture.
    pub fn latest_alias(self) -> &'static str {
        match self {
            Self::ParakeetTdt => "parakeet-latest",
        }
    }
}

/// Whether the checkpoint `config.json` describes a speech model this binary knows.
pub(crate) fn is_speech_checkpoint(config: &serde_json::Value) -> bool {
    config["model_type"].as_str() == Some("parakeet_tdt")
}

/// The speech architecture of the checkpoint directory at `path`.
pub fn detect_speech(path: &Path) -> Result<SpeechArchitecture> {
    let dir = checkpoint_dir(path).ok_or_else(|| {
        Error::UnsupportedModel(format!(
            "{} is not a Hugging Face checkpoint directory",
            path.display()
        ))
    })?;
    let config = read_config(&dir)?;
    if is_speech_checkpoint(&config) {
        return Ok(SpeechArchitecture::ParakeetTdt);
    }
    Err(Error::UnsupportedModel(format!(
        "checkpoint model_type {:?} is not a supported speech model",
        config["model_type"].as_str().unwrap_or_default()
    )))
}

fn read_config(dir: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|_| Error::ModelLoad)?;
    serde_json::from_str(&text).map_err(|_| Error::ModelLoad)
}

/// The model ID served by default for the speech model at `path`: `parakeet-tdt-0.6b-v3` for
/// the 24-layer multilingual checkpoint, otherwise the architecture ID.
pub fn default_speech_model_id(path: &Path, architecture: SpeechArchitecture) -> String {
    let config = checkpoint_dir(path).and_then(|dir| read_config(&dir).ok());
    let v3 = config.is_some_and(|c| {
        c["vocab_size"].as_u64() == Some(8193)
            && c["encoder_config"]["num_hidden_layers"].as_u64() == Some(24)
    });
    match architecture {
        SpeechArchitecture::ParakeetTdt if v3 => "parakeet-tdt-0.6b-v3".into(),
        other => other.id().into(),
    }
}

/// Loads the speech model named by `config` on the calling thread.
pub fn load_speech(config: &SpeechConfig) -> Result<Box<dyn SpeechModel>> {
    let architecture = detect_speech(&config.model)?;
    match architecture {
        #[cfg(feature = "parakeet")]
        SpeechArchitecture::ParakeetTdt => {
            let mut config = config.clone();
            config.model = checkpoint_dir(&config.model).unwrap_or(config.model);
            Ok(Box::new(jevons_parakeet::Parakeet::load(&config)?))
        }
        #[allow(unreachable_patterns)]
        other => Err(Error::UnsupportedModel(format!(
            "{} support is not built into this binary",
            other.id()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_checkpoints_are_speech_models_and_llms_are_not() {
        let dir = std::env::temp_dir().join(format!("jevons-speech-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"parakeet_tdt","vocab_size":8193,"encoder_config":{"num_hidden_layers":24}}"#,
        )
        .unwrap();
        let architecture = detect_speech(&dir).unwrap();
        assert_eq!(architecture, SpeechArchitecture::ParakeetTdt);
        assert_eq!(
            default_speech_model_id(&dir, architecture),
            "parakeet-tdt-0.6b-v3"
        );
        assert!(matches!(
            crate::detect(&dir),
            Err(Error::UnsupportedModel(message)) if message.contains("--speech-model")
        ));
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"nemotron_labs_diffusion"}"#,
        )
        .unwrap();
        assert!(matches!(
            detect_speech(&dir),
            Err(Error::UnsupportedModel(_))
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
