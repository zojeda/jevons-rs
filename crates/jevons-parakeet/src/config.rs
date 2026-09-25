//! `config.json` and `processor_config.json` of a Hugging Face Parakeet TDT checkpoint,
//! validated against what this runtime implements.
use jevons_audio::MelConfig;
use jevons_core::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_mel_bins: usize,
    pub conv_kernel_size: usize,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub convolution_bias: bool,
    pub scale_input: bool,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
    pub max_position_embeddings: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub encoder_config: EncoderConfig,
    pub vocab_size: usize,
    pub blank_token_id: usize,
    pub decoder_hidden_size: usize,
    pub num_decoder_layers: usize,
    pub durations: Vec<usize>,
    pub hidden_act: String,
    pub max_symbols_per_step: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct FeatureExtractor {
    feature_size: usize,
    sampling_rate: u32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    preemphasis: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct Processor {
    feature_extractor: FeatureExtractor,
}

fn unsupported(message: String) -> Error {
    Error::UnsupportedModel(format!("Parakeet TDT config: {message}"))
}

impl Config {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| unsupported(format!("cannot read config.json: {e}")))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text)
            .map_err(|e| unsupported(format!("invalid config.json: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let e = &self.encoder_config;
        let heads = e.num_attention_heads;
        let checks = [
            (
                self.model_type == "parakeet_tdt",
                format!("model_type {:?}", self.model_type),
            ),
            (
                e.hidden_act == "silu" && self.hidden_act == "relu",
                format!("activations {:?} / {:?}", e.hidden_act, self.hidden_act),
            ),
            (
                !e.attention_bias && !e.convolution_bias && !e.scale_input,
                "attention or convolution biases, or input scaling".into(),
            ),
            (
                heads > 0 && e.num_key_value_heads == heads && e.hidden_size.is_multiple_of(heads),
                format!("{heads} heads, {} KV heads", e.num_key_value_heads),
            ),
            (
                e.subsampling_conv_kernel_size == 3
                    && e.subsampling_conv_stride == 2
                    && e.subsampling_factor.is_power_of_two()
                    && e.subsampling_factor >= 2
                    && e.num_mel_bins.is_multiple_of(e.subsampling_factor),
                format!(
                    "subsampling kernel {} stride {} factor {}",
                    e.subsampling_conv_kernel_size, e.subsampling_conv_stride, e.subsampling_factor
                ),
            ),
            (
                e.conv_kernel_size % 2 == 1,
                format!("even convolution kernel {}", e.conv_kernel_size),
            ),
            (
                self.blank_token_id + 1 == self.vocab_size,
                format!(
                    "blank token {} of vocabulary {}",
                    self.blank_token_id, self.vocab_size
                ),
            ),
            (
                !self.durations.is_empty()
                    && self.durations.iter().enumerate().all(|(i, &d)| i == d),
                format!("durations {:?}", self.durations),
            ),
            (
                self.num_decoder_layers > 0 && self.max_symbols_per_step > 0,
                "empty prediction network".into(),
            ),
        ];
        match checks.into_iter().find(|(ok, _)| !ok) {
            Some((_, message)) => Err(unsupported(format!("unsupported {message}"))),
            None => Ok(()),
        }
    }

    /// Encoder frames of `mel` feature frames: each stride-2 convolution maps `n` to
    /// `(n + 2·1 - 3) / 2 + 1`.
    pub fn encoded_frames(&self, mel: usize) -> usize {
        let layers = self.encoder_config.subsampling_factor.trailing_zeros();
        (0..layers).fold(mel, |n, _| n.div_ceil(2))
    }
}

/// The feature extractor of `processor_config.json`.
pub fn mel_config(dir: &Path, config: &Config) -> Result<MelConfig> {
    let text = std::fs::read_to_string(dir.join("processor_config.json"))
        .map_err(|e| unsupported(format!("cannot read processor_config.json: {e}")))?;
    parse_mel(&text, config)
}

fn parse_mel(text: &str, config: &Config) -> Result<MelConfig> {
    let processor: Processor = serde_json::from_str(text)
        .map_err(|e| unsupported(format!("invalid processor_config.json: {e}")))?;
    let f = processor.feature_extractor;
    if f.feature_size != config.encoder_config.num_mel_bins
        || f.win_length > f.n_fft
        || f.hop_length == 0
    {
        return Err(unsupported(format!(
            "feature extractor with {} mels, window {}, FFT {}, hop {}",
            f.feature_size, f.win_length, f.n_fft, f.hop_length
        )));
    }
    Ok(MelConfig {
        sample_rate: f.sampling_rate,
        n_fft: f.n_fft,
        win_length: f.win_length,
        hop_length: f.hop_length,
        n_mels: f.feature_size,
        preemphasis: f.preemphasis,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `nvidia/parakeet-tdt-0.6b-v3` `config.json`.
    pub(crate) const V3: &str = r#"{
  "architectures": ["ParakeetForTDT"],
  "blank_token_id": 8192,
  "decoder_hidden_size": 640,
  "dtype": "float32",
  "durations": [0, 1, 2, 3, 4],
  "encoder_config": {
    "activation_dropout": 0.1, "attention_bias": false, "attention_dropout": 0.1,
    "conv_kernel_size": 9, "convolution_bias": false, "dropout": 0.1, "dropout_positions": 0.0,
    "hidden_act": "silu", "hidden_size": 1024, "initializer_range": 0.02,
    "intermediate_size": 4096, "layerdrop": 0.1, "max_position_embeddings": 5000,
    "model_type": "parakeet_encoder", "num_attention_heads": 8, "num_hidden_layers": 24,
    "num_key_value_heads": 8, "num_mel_bins": 128, "scale_input": false,
    "subsampling_conv_channels": 256, "subsampling_conv_kernel_size": 3,
    "subsampling_conv_stride": 2, "subsampling_factor": 8
  },
  "hidden_act": "relu",
  "initializer_range": 0.02,
  "is_encoder_decoder": true,
  "max_symbols_per_step": 10,
  "model_type": "parakeet_tdt",
  "num_decoder_layers": 2,
  "pad_token_id": 2,
  "transformers_version": "5.6.0.dev0",
  "vocab_size": 8193
}"#;

    /// `nvidia/parakeet-tdt-0.6b-v3` `processor_config.json`.
    const PROCESSOR: &str = r#"{
  "blank_token": "<blank>",
  "feature_extractor": {
    "feature_extractor_type": "ParakeetFeatureExtractor", "feature_size": 128,
    "hop_length": 160, "n_fft": 512, "padding_side": "right", "padding_value": 0.0,
    "preemphasis": 0.97, "return_attention_mask": true, "sampling_rate": 16000,
    "win_length": 400
  },
  "processor_class": "ParakeetProcessor"
}"#;

    #[test]
    fn the_published_v3_configs_parse() {
        let config = Config::parse(V3).unwrap();
        assert_eq!(config.encoder_config.num_hidden_layers, 24);
        assert_eq!(config.encoded_frames(1600), 200);
        assert_eq!(config.encoded_frames(585), 74);
        let mel = parse_mel(PROCESSOR, &config).unwrap();
        assert_eq!((mel.n_fft, mel.win_length, mel.hop_length), (512, 400, 160));
    }

    #[test]
    fn unsupported_variants_are_rejected() {
        for (from, to) in [
            (
                r#""model_type": "parakeet_tdt""#,
                r#""model_type": "parakeet_ctc""#,
            ),
            (r#""attention_bias": false"#, r#""attention_bias": true"#),
            (r#""scale_input": false"#, r#""scale_input": true"#),
            (
                r#""durations": [0, 1, 2, 3, 4]"#,
                r#""durations": [1, 2, 4]"#,
            ),
            (r#""blank_token_id": 8192"#, r#""blank_token_id": 0"#),
            (r#""num_key_value_heads": 8"#, r#""num_key_value_heads": 4"#),
            (r#""subsampling_factor": 8"#, r#""subsampling_factor": 6"#),
        ] {
            let text = V3.replace(from, to);
            assert_ne!(text, V3, "{from}");
            assert!(
                matches!(Config::parse(&text), Err(Error::UnsupportedModel(_))),
                "{to} was accepted"
            );
        }
    }
}
