//! `config.json` and the speech tokenizer's `config.json`, and the variants this runtime
//! accepts.
use jevons_core::{Error, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// ISO-639-1 codes and the names the checkpoint's `codec_language_id` gives its languages.
pub const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("zh", "chinese"),
    ("en", "english"),
    ("de", "german"),
    ("it", "italian"),
    ("pt", "portuguese"),
    ("es", "spanish"),
    ("ja", "japanese"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ru", "russian"),
];

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub tts_model_type: String,
    pub tokenizer_type: String,
    pub im_start_token_id: i32,
    pub im_end_token_id: i32,
    pub assistant_token_id: i32,
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    pub speaker_encoder_config: SpeakerConfig,
    pub talker_config: TalkerConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpeakerConfig {
    /// Width of a speaker vector.
    pub enc_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    /// Codec vocabulary of the first codebook, with the control ids above the codes.
    pub vocab_size: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    /// Codebooks per audio frame.
    pub num_code_groups: usize,
    pub codec_bos_id: i32,
    pub codec_eos_token_id: i32,
    pub codec_pad_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    pub codec_language_id: BTreeMap<String, i32>,
    pub code_predictor_config: PredictorConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PredictorConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub vocab_size: usize,
    pub num_code_groups: usize,
}

/// The speech tokenizer's `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct CodecConfig {
    pub output_sample_rate: u32,
    /// Samples per frame.
    pub decode_upsample_rate: usize,
    pub decoder_config: DecoderConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub hidden_act: String,
    pub attention_bias: bool,
    /// Strides of the four vocoder stages.
    pub upsample_rates: Vec<usize>,
    /// Strides of the stages between the transformer and the vocoder.
    pub upsampling_ratios: Vec<usize>,
}

fn unsupported(what: impl std::fmt::Display) -> Error {
    Error::UnsupportedModel(format!("Qwen3-TTS: {what}"))
}

fn read<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|_| Error::ModelLoad)?;
    serde_json::from_str(&text).map_err(|e| unsupported(format!("{}: {e}", path.display())))
}

impl Config {
    /// Reads and checks `config.json` and `speech_tokenizer/config.json` in `dir`.
    pub fn from_dir(dir: &Path) -> Result<(Self, CodecConfig)> {
        let config: Self = read(&dir.join("config.json"))?;
        let codec: CodecConfig = read(&dir.join("speech_tokenizer").join("config.json"))?;
        config.validate(&codec)?;
        Ok((config, codec))
    }

    fn validate(&self, codec: &CodecConfig) -> Result<()> {
        if self.tts_model_type != "base" || self.tokenizer_type != "qwen3_tts_tokenizer_12hz" {
            return Err(unsupported(format!(
                "{:?} with {:?} is not a 12Hz Base checkpoint",
                self.tts_model_type, self.tokenizer_type
            )));
        }
        let t = &self.talker_config;
        let p = &t.code_predictor_config;
        let d = &codec.decoder_config;
        if t.hidden_act != "silu" || p.hidden_act != "silu" || d.hidden_act != "silu" {
            return Err(unsupported("only silu networks are supported"));
        }
        if t.attention_bias || p.attention_bias || d.attention_bias {
            return Err(unsupported("attention biases are not supported"));
        }
        // The 1.7B talker is wider than its predictor and projects between them.
        if p.hidden_size != t.hidden_size {
            return Err(unsupported(format!(
                "a talker of width {} over a predictor of width {} is not supported",
                t.hidden_size, p.hidden_size
            )));
        }
        for (name, heads, kv_heads, head_dim, hidden, intermediate) in [
            (
                "talker",
                t.num_attention_heads,
                t.num_key_value_heads,
                t.head_dim,
                t.hidden_size,
                t.intermediate_size,
            ),
            (
                "code predictor",
                p.num_attention_heads,
                p.num_key_value_heads,
                p.head_dim,
                p.hidden_size,
                p.intermediate_size,
            ),
            (
                "codec",
                d.num_attention_heads,
                d.num_key_value_heads,
                d.head_dim,
                d.hidden_size,
                d.intermediate_size,
            ),
        ] {
            let whole = |n: usize| n > 0 && n.is_multiple_of(64);
            if kv_heads == 0
                || !heads.is_multiple_of(kv_heads)
                || !head_dim.is_multiple_of(2)
                || !whole(hidden)
                || !whole(intermediate)
                || !whole(heads * head_dim)
                || !whole(kv_heads * head_dim)
            {
                return Err(unsupported(format!("the {name}'s shape is not supported")));
            }
        }
        if !t.vocab_size.is_multiple_of(64)
            || !p.vocab_size.is_multiple_of(64)
            || !t.text_hidden_size.is_multiple_of(64)
        {
            return Err(unsupported("vocabulary sizes must be multiples of 64"));
        }
        if t.num_code_groups < 2
            || p.num_code_groups != t.num_code_groups
            || d.num_quantizers != t.num_code_groups
            || d.num_semantic_quantizers != 1
        {
            return Err(unsupported("the codebook counts do not agree"));
        }
        if self.speaker_encoder_config.enc_dim != t.hidden_size {
            return Err(unsupported(
                "a speaker vector must be as wide as the talker",
            ));
        }
        let control = [
            t.codec_bos_id,
            t.codec_eos_token_id,
            t.codec_pad_id,
            t.codec_think_id,
            t.codec_nothink_id,
            t.codec_think_bos_id,
            t.codec_think_eos_id,
        ];
        let in_vocab = |id: i32| usize::try_from(id).is_ok_and(|id| id < t.vocab_size);
        if !control.iter().copied().all(in_vocab)
            || !t.codec_language_id.values().copied().all(in_vocab)
            || p.vocab_size > t.vocab_size
        {
            return Err(unsupported("a control id is outside the codec vocabulary"));
        }
        if let Some((_, name)) = LANGUAGE_NAMES
            .iter()
            .find(|(_, name)| !t.codec_language_id.contains_key(*name))
        {
            return Err(unsupported(format!(
                "the checkpoint names no {name} language"
            )));
        }
        let stride: usize = d
            .upsample_rates
            .iter()
            .chain(&d.upsampling_ratios)
            .product();
        if d.upsample_rates.len() != 4
            || d.upsample_rates.contains(&0)
            || d.upsampling_ratios.contains(&0)
            || stride != codec.decode_upsample_rate
            || !d.decoder_dim.is_multiple_of(16)
            || !d.codebook_dim.is_multiple_of(2)
            || d.sliding_window == 0
        {
            return Err(unsupported("the codec decoder's shape is not supported"));
        }
        Ok(())
    }

    /// The codec id that names `language` (ISO-639-1) in a prompt.
    pub fn language_id(&self, language: &str) -> Option<i32> {
        let (_, name) = LANGUAGE_NAMES.iter().find(|(code, _)| *code == language)?;
        self.talker_config.codec_language_id.get(*name).copied()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The 0.6B Base checkpoint's configuration, cut to what the runtime reads.
    pub(crate) const CONFIG: &str = r#"{
      "tts_model_type": "base", "tokenizer_type": "qwen3_tts_tokenizer_12hz",
      "im_start_token_id": 151644, "im_end_token_id": 151645, "assistant_token_id": 77091,
      "tts_bos_token_id": 151672, "tts_eos_token_id": 151673, "tts_pad_token_id": 151671,
      "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
      "talker_config": {
        "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
        "rms_norm_eps": 1e-06, "rope_theta": 1000000, "hidden_act": "silu",
        "attention_bias": false, "vocab_size": 3072, "text_hidden_size": 2048,
        "text_vocab_size": 151936, "num_code_groups": 16,
        "codec_bos_id": 2149, "codec_eos_token_id": 2150, "codec_pad_id": 2148,
        "codec_think_id": 2154, "codec_nothink_id": 2155, "codec_think_bos_id": 2156,
        "codec_think_eos_id": 2157,
        "codec_language_id": {
          "chinese": 2055, "english": 2050, "german": 2053, "italian": 2070, "portuguese": 2071,
          "spanish": 2054, "japanese": 2058, "korean": 2064, "french": 2061, "russian": 2069
        },
        "code_predictor_config": {
          "hidden_size": 1024, "num_hidden_layers": 5, "num_attention_heads": 16,
          "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
          "rms_norm_eps": 1e-06, "rope_theta": 1000000, "hidden_act": "silu",
          "attention_bias": false, "vocab_size": 2048, "num_code_groups": 16
        }
      }
    }"#;

    pub(crate) const CODEC: &str = r#"{
      "output_sample_rate": 24000, "decode_upsample_rate": 1920,
      "decoder_config": {
        "latent_dim": 1024, "codebook_dim": 512, "codebook_size": 2048, "decoder_dim": 1536,
        "hidden_size": 512, "intermediate_size": 1024, "head_dim": 64,
        "num_attention_heads": 16, "num_key_value_heads": 16, "num_hidden_layers": 8,
        "num_quantizers": 16, "num_semantic_quantizers": 1, "rms_norm_eps": 1e-05,
        "rope_theta": 10000, "sliding_window": 72, "hidden_act": "silu",
        "attention_bias": false, "upsample_rates": [8, 5, 4, 3], "upsampling_ratios": [2, 2]
      }
    }"#;

    fn parsed() -> (Config, CodecConfig) {
        (
            serde_json::from_str(CONFIG).unwrap(),
            serde_json::from_str(CODEC).unwrap(),
        )
    }

    #[test]
    fn the_base_checkpoint_is_accepted_and_names_its_languages() {
        let (config, codec) = parsed();
        config.validate(&codec).unwrap();
        assert_eq!(config.language_id("es"), Some(2054));
        assert_eq!(config.language_id("en"), Some(2050));
        assert_eq!(config.language_id("nl"), None);
        assert_eq!(config.language_id("spanish"), None);
    }

    #[test]
    fn other_variants_are_refused_by_name() {
        let refused = |edit: &dyn Fn(&mut Config, &mut CodecConfig)| {
            let (mut config, mut codec) = parsed();
            edit(&mut config, &mut codec);
            match config.validate(&codec) {
                Err(Error::UnsupportedModel(message)) => message,
                other => panic!("accepted: {other:?}"),
            }
        };
        assert!(refused(&|c, _| c.tts_model_type = "custom_voice".into()).contains("Base"));
        assert!(refused(&|c, _| c.talker_config.hidden_size = 2048).contains("predictor of width"));
        assert!(refused(&|c, _| c.talker_config.hidden_act = "gelu".into()).contains("silu"));
        assert!(refused(&|c, _| c.talker_config.attention_bias = true).contains("biases"));
        assert!(refused(&|_, k| k.decoder_config.num_quantizers = 8).contains("codebook counts"));
        assert!(refused(&|_, k| k.decode_upsample_rate = 2000).contains("codec decoder"));
        assert!(refused(&|c, _| c.talker_config.codec_eos_token_id = 4000).contains("control id"));
        assert!(refused(&|c, _| c.speaker_encoder_config.enc_dim = 512).contains("speaker"));
        assert!(
            refused(&|c, _| {
                c.talker_config.codec_language_id.remove("korean");
            })
            .contains("korean")
        );
    }
}
