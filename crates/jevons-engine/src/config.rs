//! Model loading and context configuration.

use crate::{Error, Result};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Reuse resident prompt KV for the longest common token prefix across prefills.
    pub prompt_cache: bool,
    pub model: PathBuf,
    /// Vision projector GGUF; enables image input.
    pub mmproj: Option<PathBuf>,
    /// HIP device index.
    pub main_gpu: usize,
    /// Positions reserved in the KV cache for prompt, thought and canvas.
    pub context_size: u32,
    /// Maximum tokens per prefill chunk and canvas.
    pub batch_size: u32,
}

impl ModelConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            prompt_cache: true,
            model: model.into(),
            mmproj: None,
            main_gpu: 0,
            context_size: 8192,
            batch_size: 512,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.context_size == 0
            || self.context_size > i32::MAX as u32
            || self.batch_size == 0
            || self.batch_size > self.context_size
        {
            return Err(Error::InvalidInput(
                "Require 0 < batch_size <= context_size <= i32::MAX".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_must_fit_in_a_nonempty_context() {
        assert!(ModelConfig::new("m.gguf").validate().is_ok());
        let mut config = ModelConfig::new("m.gguf");
        config.batch_size = config.context_size + 1;
        assert!(config.validate().is_err());
        config.batch_size = 0;
        assert!(config.validate().is_err());
    }
}
