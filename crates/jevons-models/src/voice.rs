//! Text-to-speech architecture detection and loading.
use crate::detect::checkpoint_dir;
use jevons_core::{Error, Result, VoiceConfig, VoiceModel};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceArchitecture {
    /// Alibaba Qwen3-TTS Base checkpoint directory with its 12 Hz speech tokenizer.
    Qwen3Tts,
}

impl VoiceArchitecture {
    pub fn id(self) -> &'static str {
        match self {
            Self::Qwen3Tts => "qwen3-tts",
        }
    }

    /// Routing alias that names the newest local model of this architecture.
    pub fn latest_alias(self) -> &'static str {
        match self {
            Self::Qwen3Tts => "qwen3-tts-latest",
        }
    }
}

/// Whether the checkpoint `config.json` describes a voice model this binary knows.
pub(crate) fn is_voice_checkpoint(config: &serde_json::Value) -> bool {
    config["model_type"].as_str() == Some("qwen3_tts")
}

/// The voice architecture of the checkpoint directory at `path`.
pub fn detect_voice(path: &Path) -> Result<VoiceArchitecture> {
    let dir = checkpoint_dir(path).ok_or_else(|| {
        Error::UnsupportedModel(format!(
            "{} is not a Hugging Face checkpoint directory",
            path.display()
        ))
    })?;
    let config = read_config(&dir)?;
    if !is_voice_checkpoint(&config) {
        return Err(Error::UnsupportedModel(format!(
            "checkpoint model_type {:?} is not a supported voice model",
            config["model_type"].as_str().unwrap_or_default()
        )));
    }
    // A CustomVoice or VoiceDesign checkpoint takes its voice another way than a Base one.
    let kind = config["tts_model_type"].as_str().unwrap_or_default();
    let tokenizer = config["tokenizer_type"].as_str().unwrap_or_default();
    if kind != "base" || tokenizer != "qwen3_tts_tokenizer_12hz" {
        return Err(Error::UnsupportedModel(format!(
            "Qwen3-TTS {kind:?} with {tokenizer:?} is not supported: use a 12Hz Base checkpoint"
        )));
    }
    Ok(VoiceArchitecture::Qwen3Tts)
}

fn read_config(dir: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join("config.json")).map_err(|_| Error::ModelLoad)?;
    serde_json::from_str(&text).map_err(|_| Error::ModelLoad)
}

/// The model ID served by default for the voice model at `path`: `qwen3-tts-12hz-0.6b-base` or
/// `qwen3-tts-12hz-1.7b-base` by the checkpoint's size, otherwise the architecture ID.
pub fn default_voice_model_id(path: &Path, architecture: VoiceArchitecture) -> String {
    let config = checkpoint_dir(path).and_then(|dir| read_config(&dir).ok());
    let size = config
        .as_ref()
        .and_then(|c| c["tts_model_size"].as_str())
        .unwrap_or_default();
    match (architecture, size) {
        (VoiceArchitecture::Qwen3Tts, "0b6") => "qwen3-tts-12hz-0.6b-base".into(),
        (VoiceArchitecture::Qwen3Tts, "1b7") => "qwen3-tts-12hz-1.7b-base".into(),
        (other, _) => other.id().into(),
    }
}

/// Loads the voice model named by `config` on the calling thread.
pub fn load_voice(config: &VoiceConfig) -> Result<Box<dyn VoiceModel>> {
    let architecture = detect_voice(&config.model)?;
    match architecture {
        #[cfg(feature = "qwen3-tts")]
        VoiceArchitecture::Qwen3Tts => {
            let mut config = config.clone();
            config.model = checkpoint_dir(&config.model).unwrap_or(config.model);
            Ok(Box::new(jevons_qwen3_tts::Qwen3Tts::load(&config)?))
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
    fn qwen3_tts_base_checkpoints_are_voice_models_and_others_are_not() {
        let dir = std::env::temp_dir().join(format!("jevons-voice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |text: &str| std::fs::write(dir.join("config.json"), text).unwrap();
        write(
            r#"{"model_type":"qwen3_tts","tts_model_type":"base","tts_model_size":"0b6","tokenizer_type":"qwen3_tts_tokenizer_12hz"}"#,
        );
        let architecture = detect_voice(&dir).unwrap();
        assert_eq!(architecture, VoiceArchitecture::Qwen3Tts);
        assert_eq!(architecture.latest_alias(), "qwen3-tts-latest");
        assert_eq!(
            default_voice_model_id(&dir, architecture),
            "qwen3-tts-12hz-0.6b-base"
        );
        // A language service is told where the checkpoint belongs, and so is a speech service.
        assert!(matches!(
            crate::detect(&dir),
            Err(Error::UnsupportedModel(message)) if message.contains("services.voice")
        ));
        assert!(matches!(
            crate::detect_speech(&dir),
            Err(Error::UnsupportedModel(message)) if message.contains("services.voice")
        ));
        write(
            r#"{"model_type":"qwen3_tts","tts_model_type":"custom_voice","tokenizer_type":"qwen3_tts_tokenizer_12hz"}"#,
        );
        assert!(matches!(
            detect_voice(&dir),
            Err(Error::UnsupportedModel(message)) if message.contains("Base checkpoint")
        ));
        write(r#"{"model_type":"parakeet_tdt"}"#);
        assert!(matches!(
            detect_voice(&dir),
            Err(Error::UnsupportedModel(_))
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
