//! Parakeet TDT on Burn: log-mel features on the host, the FastConformer encoder on the GPU,
//! then greedy token-and-duration decoding.
//!
//! Tensor names follow the Hugging Face `ParakeetForTDT` layout: `encoder.subsampling.*`,
//! `encoder.layers.{l}.{feed_forward1,self_attn,conv,feed_forward2,norm_*}.*`,
//! `encoder_projector.*`, `decoder.{embedding,lstm,decoder_projector}.*` and `joint.head.*`.
use crate::config::{Config, mel_config};
use crate::decoder::Decoder;
use crate::encoder::Encoder;
use jevons_audio::LogMel;
use jevons_burn::weights::Loader;
use jevons_burn::{DType, Device, Tensor, TensorData};
use jevons_core::{
    Error, Result, SpeechConfig, SpeechInfo, SpeechModel, SpeechToken, TextTokenizer,
};
use jevons_formats::safetensors::Checkpoint;
use jevons_tokenizer::hf::HfTokenizer;

/// The 25 languages of Parakeet TDT 0.6B v3, which it detects by itself.
pub const LANGUAGES: &[&str] = &[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

/// Longest audio one pass attends over; longer audio is windowed by the caller.
const MAX_WINDOW_SECONDS: f64 = 120.0;
/// Encoder frames are padded to a multiple of this (5.12 s), capping compiled shapes.
const FRAME_BUCKET: usize = 64;

pub struct Parakeet {
    config: Config,
    device: Device,
    mel: LogMel,
    tokenizer: HfTokenizer,
    info: SpeechInfo,
    encoder: Encoder,
    decoder: Decoder,
}

fn load_error(error: impl std::fmt::Display) -> Error {
    Error::UnsupportedModel(format!("Parakeet TDT weights: {error}"))
}

impl Parakeet {
    /// Loads a checkpoint directory with `config.json`, `processor_config.json`,
    /// `tokenizer.json` and safetensors weights.
    pub fn load(config: &SpeechConfig) -> Result<Self> {
        let dir = &config.model;
        let model_config = Config::from_dir(dir)?;
        let mel = mel_config(dir, &model_config)?;
        let tokenizer = HfTokenizer::from_file(&dir.join("tokenizer.json"), None)?;
        // The tokenizer lists `<blank>` as a special added token.
        if tokenizer.n_vocab() != model_config.vocab_size {
            return Err(Error::UnsupportedModel(format!(
                "Parakeet TDT tokenizer has {} tokens, the model {}",
                tokenizer.n_vocab(),
                model_config.vocab_size
            )));
        }
        let checkpoint = Checkpoint::open_dir(dir).map_err(load_error)?;
        let device = jevons_burn::device::hip(config.main_gpu);
        let load = Loader {
            checkpoint: &checkpoint,
            device: &device,
        };
        let encoder = Encoder::load(&load, &model_config.encoder_config).map_err(load_error)?;
        let decoder = Decoder::load(&load, &model_config).map_err(load_error)?;
        device.memory_cleanup();
        let e = &model_config.encoder_config;
        let info = SpeechInfo {
            architecture: "parakeet-tdt",
            display_name: format!(
                "Parakeet TDT ({} layers, {} hidden)",
                e.num_hidden_layers, e.hidden_size
            ),
            sample_rate: mel.sample_rate,
            frame_seconds: (mel.hop_length * e.subsampling_factor) as f64
                / f64::from(mel.sample_rate),
            max_window_seconds: MAX_WINDOW_SECONDS,
            languages: LANGUAGES,
        };
        Ok(Self {
            config: model_config,
            device,
            mel: LogMel::new(mel),
            tokenizer,
            info,
            encoder,
            decoder,
        })
    }

    /// Encoder rows of `samples`, padded to a frame bucket, and the valid row count.
    fn encode(&mut self, samples: &[f32]) -> (Tensor<2>, usize) {
        let features = self.mel.features(samples);
        let factor = self.config.encoder_config.subsampling_factor;
        let frames = self
            .config
            .encoded_frames(features.frames)
            .next_multiple_of(FRAME_BUCKET);
        let mut data = features.data;
        data.resize(frames * factor * features.n_mels, 0.0);
        let mel = Tensor::<2>::from_data(
            TensorData::new(data, [frames * factor, features.n_mels]),
            (&self.device, DType::F32),
        );
        let (rows, valid) = self.encoder.subsample(mel, features.valid);
        (self.encoder.blocks(rows, valid), valid)
    }
}

impl SpeechModel for Parakeet {
    fn info(&self) -> &SpeechInfo {
        &self.info
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<SpeechToken>> {
        let rate = f64::from(self.info.sample_rate);
        if samples.len() as f64 > self.info.max_window_seconds * rate {
            return Err(Error::InvalidInput(format!(
                "A transcription window is at most {} seconds",
                self.info.max_window_seconds
            )));
        }
        if samples.len() < self.mel.config().hop_length {
            return Ok(Vec::new());
        }
        let (encoded, valid) = self.encode(samples);
        let projected = self.decoder.project(encoded);
        let seconds = self.info.frame_seconds;
        let tokens = self
            .decoder
            .decode(projected, valid)
            .into_iter()
            .map(|e| SpeechToken {
                id: e.token as u32,
                piece: self.tokenizer.piece(e.token as u32).unwrap_or_default(),
                start: e.frame as f64 * seconds,
                end: (e.frame + e.frames) as f64 * seconds,
                logprob: e.logprob,
            })
            .collect();
        Ok(tokens)
    }

    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        self.tokenizer.decode(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_burn::layers::host_f32;
    use std::path::PathBuf;

    fn golden_dir() -> PathBuf {
        std::env::var_os("JEVONS_GOLDEN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/jevons/golden")
            })
            .join(
                std::env::var("PARAKEET_GOLDEN").unwrap_or_else(|_| "parakeet-tdt-0.6b-v3".into()),
            )
    }

    fn golden(name: &str) -> Vec<u8> {
        let path = golden_dir().join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn floats(name: &str) -> Vec<f32> {
        golden(&format!("{name}.f32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn ints(name: &str) -> Vec<i32> {
        golden(&format!("{name}.i32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }

    fn manifest() -> serde_json::Value {
        serde_json::from_slice(&golden("manifest.json")).unwrap()
    }

    fn load_model() -> Parakeet {
        Parakeet::load(&SpeechConfig::new(std::env::var("PARAKEET_MODEL").unwrap())).unwrap()
    }

    /// RMS of `got - want` relative to the RMS of `want`, and the largest difference.
    fn relative_error(got: &[f32], want: &[f32]) -> (f32, f32) {
        assert_eq!(got.len(), want.len());
        let rms = |v: &mut dyn Iterator<Item = f32>| {
            let (sum, n) = v.fold((0f64, 0usize), |(s, n), x| {
                (s + f64::from(x).powi(2), n + 1)
            });
            (sum / n as f64).sqrt() as f32
        };
        let error = rms(&mut got.iter().zip(want).map(|(a, b)| a - b));
        let scale = rms(&mut want.iter().copied());
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        (error / scale, worst)
    }

    #[test]
    #[ignore = "Requires PARAKEET_MODEL, a HIP GPU and the reference dump"]
    fn encoder_matches_the_reference() {
        let mut model = load_model();
        let d = model.config.encoder_config.hidden_size;
        let traced: Vec<usize> = manifest()["traced_layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        for clip in ["en", "es"] {
            let audio = floats(&format!("{clip}_audio"));
            let features = model.mel.features(&audio);
            let factor = model.config.encoder_config.subsampling_factor;
            let frames = model
                .config
                .encoded_frames(features.frames)
                .next_multiple_of(FRAME_BUCKET);
            let mut data = features.data.clone();
            data.resize(frames * factor * features.n_mels, 0.0);
            let mel = Tensor::<2>::from_data(
                TensorData::new(data, [frames * factor, features.n_mels]),
                (&model.device, DType::F32),
            );
            let (rows, valid) = model.encoder.subsample(mel, features.valid);
            let reference = floats(&format!("{clip}_subsampling"));
            // The reference keeps its own padding frame; compare the valid rows.
            let (error, worst) = relative_error(
                &host_f32(rows.clone().slice([0..valid, 0..d])),
                &reference[..valid * d],
            );
            println!("{clip} subsampling: relative RMS error {error:.2e}, max {worst:.2e}");
            assert!(error < 1e-2, "{clip} subsampling");

            model.encoder.trace = Some(Vec::new());
            let encoded = model.encoder.blocks(rows, valid);
            let trace = model.encoder.trace.take().unwrap();
            for &layer in &traced {
                let reference = floats(&format!("{clip}_layer{layer}"));
                let (error, worst) = relative_error(&trace[layer], &reference[..valid * d]);
                println!("{clip} layer {layer}: relative RMS error {error:.2e}, max {worst:.2e}");
            }
            let reference = floats(&format!("{clip}_encoder"));
            let got = host_f32(encoded.clone().slice([0..valid, 0..d]));
            let (error, worst) = relative_error(&got, &reference[..valid * d]);
            println!("{clip} encoder: relative RMS error {error:.2e}, max {worst:.2e}");
            assert!(error < 3e-2, "{clip} encoder");

            let h = model.config.decoder_hidden_size;
            let projected = host_f32(model.decoder.project(encoded).slice([0..valid, 0..h]));
            let reference = floats(&format!("{clip}_projected"));
            let (error, _) = relative_error(&projected, &reference[..valid * h]);
            println!("{clip} projected: relative RMS error {error:.2e}");
            assert!(error < 3e-2, "{clip} projected");
        }
    }

    #[test]
    #[ignore = "Requires PARAKEET_MODEL, a HIP GPU and the reference dump"]
    fn greedy_transcripts_match_the_reference() {
        let mut model = load_model();
        let manifest = manifest();
        for clip in ["en", "es"] {
            let audio = floats(&format!("{clip}_audio"));
            let start = std::time::Instant::now();
            let tokens = model.transcribe(&audio).unwrap();
            let elapsed = start.elapsed();
            let ids: Vec<i32> = tokens.iter().map(|t| t.id as i32).collect();
            let frames: Vec<i32> = tokens
                .iter()
                .map(|t| (t.start / model.info.frame_seconds).round() as i32)
                .collect();
            let text = model
                .detokenize(&tokens.iter().map(|t| t.id).collect::<Vec<_>>())
                .unwrap();
            println!(
                "{clip}: {:.2} s of audio in {elapsed:?}: {text}",
                audio.len() as f64 / 16000.0
            );
            let want = manifest["clips"][clip]["text"].as_str().unwrap();
            assert_eq!(ids, ints(&format!("{clip}_tokens")), "{clip} tokens");
            assert_eq!(
                frames,
                ints(&format!("{clip}_token_frames")),
                "{clip} frames"
            );
            assert_eq!(text, want, "{clip} text");
            assert!(tokens.iter().all(|t| t.logprob <= 0.0 && t.end > t.start));
        }
    }
}
