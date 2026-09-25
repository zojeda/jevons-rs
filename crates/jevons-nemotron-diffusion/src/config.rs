//! `config.json` of a Nemotron-Labs-Diffusion checkpoint, validated against what this runtime
//! implements.
use jevons_core::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(default)]
    pub truncate: Option<bool>,
    #[serde(default)]
    pub llama_4_scaling_beta: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub mask_token_id: i32,
    pub block_size: usize,
    pub dlm_paradigm: String,
    pub eos_token_id: i32,
}

fn unsupported(message: String) -> Error {
    Error::UnsupportedModel(format!("Nemotron-Labs-Diffusion config: {message}"))
}

impl Config {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(dir.join("config.json")).map_err(|_| Error::ModelLoad)?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text).map_err(|e| unsupported(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let rope = &self.rope_parameters;
        let checks = [
            (
                matches!(
                    self.model_type.as_str(),
                    "nemotron_labs_diffusion" | "nemotron_labs_diffusion_vlm"
                ),
                format!("model_type {}", self.model_type),
            ),
            (
                self.dlm_paradigm == "bidirectional",
                format!("dlm_paradigm {}", self.dlm_paradigm),
            ),
            (
                matches!(rope.rope_type.as_str(), "yarn" | "default"),
                format!("rope_type {}", rope.rope_type),
            ),
            (
                self.sliding_window.is_none(),
                "sliding-window attention".into(),
            ),
            (
                self.hidden_act == "silu",
                format!("hidden_act {}", self.hidden_act),
            ),
            (
                !self.attention_bias && !self.mlp_bias && !self.tie_word_embeddings,
                "biases or tied embeddings".into(),
            ),
            (
                self.num_key_value_heads > 0
                    && self
                        .num_attention_heads
                        .is_multiple_of(self.num_key_value_heads)
                    && self.head_dim.is_multiple_of(2),
                "head layout".into(),
            ),
            (
                (0..self.vocab_size as i64).contains(&i64::from(self.mask_token_id)),
                format!("mask_token_id {}", self.mask_token_id),
            ),
            (self.block_size > 0, "block_size 0".into()),
        ];
        match checks.into_iter().find(|(ok, _)| !ok) {
            Some((_, what)) => Err(unsupported(format!("unsupported {what}"))),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
pub(crate) const VLM_8B: &str = r#"{
  "architectures": ["NemotronLabsDiffusionVLMModel"], "attention_bias": false, "block_size": 32,
  "dlm_paradigm": "bidirectional", "dlm_type": "llada", "eos_token_id": 11, "head_dim": 128,
  "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 14336, "mask_token_id": 100,
  "max_position_embeddings": 262144, "mlp_bias": false, "model_type": "nemotron_labs_diffusion_vlm",
  "num_attention_heads": 32, "num_hidden_layers": 34, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
  "rope_parameters": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1,
    "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
    "rope_theta": 1000000.0, "rope_type": "yarn", "type": "yarn"},
  "sliding_window": null, "tie_word_embeddings": false, "vocab_size": 131073
}"#;

/// The text-only 3B checkpoint (`nvidia/Nemotron-Labs-Diffusion-3B`).
#[cfg(test)]
pub(crate) const TEXT_3B: &str = r#"{
  "architectures": ["NemotronLabsDiffusionModel"], "attention_bias": false, "block_size": 32,
  "dlm_paradigm": "bidirectional", "eos_token_id": 11, "head_dim": 128, "hidden_act": "silu",
  "hidden_size": 3072, "intermediate_size": 9216, "mask_token_id": 100,
  "max_position_embeddings": 262144, "mlp_bias": false, "model_type": "nemotron_labs_diffusion",
  "num_attention_heads": 32, "num_hidden_layers": 26, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
  "rope_parameters": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1,
    "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
    "rope_theta": 1000000.0, "rope_type": "yarn"},
  "sliding_window": null, "tie_word_embeddings": false, "vocab_size": 131072
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_only_3b_config_parses() {
        let config = Config::parse(TEXT_3B).unwrap();
        assert_eq!(config.model_type, "nemotron_labs_diffusion");
        assert_eq!((config.num_hidden_layers, config.hidden_size), (26, 3072));
        assert_eq!(config.num_attention_heads * config.head_dim, 4096);
        let other = TEXT_3B.replace("nemotron_labs_diffusion\"", "llama\"");
        assert!(Config::parse(&other).is_err());
    }

    #[test]
    fn the_8b_config_parses_and_unsupported_variants_are_rejected() {
        let config = Config::parse(VLM_8B).unwrap();
        assert_eq!(config.num_hidden_layers, 34);
        assert_eq!(config.rope_parameters.factor, Some(16.0));
        for (from, to) in [
            (r#""sliding_window": null"#, r#""sliding_window": 4096"#),
            (
                r#""dlm_paradigm": "bidirectional""#,
                r#""dlm_paradigm": "block_diff""#,
            ),
            (r#""rope_type": "yarn""#, r#""rope_type": "longrope""#),
            (r#""mask_token_id": 100"#, r#""mask_token_id": -1"#),
            (r#""hidden_act": "silu""#, r#""hidden_act": "gelu""#),
        ] {
            let text = VLM_8B.replace(from, to);
            assert!(Config::parse(&text).is_err(), "{to}");
        }
    }
}
