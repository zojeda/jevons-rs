//! CubeCL/HIP backend adapter: Rust tokenizer, GPU model, vision encoder and prompt-prefix
//! reuse.
use crate::backend::{Backend, PromptPart};
use crate::{Error, ImageInput, ModelConfig, PrefillProfile, Result};
use jevons_cubecl::model::Segment;
use jevons_cubecl::vision::{EncodedImage, Vision};
use jevons_cubecl::{gguf::Gguf, model, tokenizer::Tokenizer, vision_input::Rgb};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

/// Image tokens per image, as in llama.cpp's DiffusionGemma integration.
const MAX_IMAGE_TOKENS: usize = 280;

struct Images {
    encoder: Vision,
    /// `<|image>` and `<image|>` delimiter tokens.
    begin: i32,
    end: i32,
    /// Images encoded for the current request, with content keys for prompt reuse.
    encoded: Vec<(EncodedImage, u64)>,
}

pub(crate) struct CubeBackend {
    model: model::Model,
    tokenizer: Tokenizer,
    profile: PrefillProfile,
    n_ctx: usize,
    batch_size: usize,
    prompt_cache: bool,
    images: Option<Images>,
}

fn map(error: model::ModelError) -> Error {
    match error {
        model::ModelError::Input(message) => Error::InvalidInput(message),
        model::ModelError::Gguf(_) => Error::ModelLoad,
        model::ModelError::Unsupported(message) => Error::Backend(message),
    }
}

impl CubeBackend {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let gguf = Gguf::open(&config.model).map_err(|_| Error::ModelLoad)?;
        let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|_| Error::UnsupportedModel)?;
        drop(gguf);
        let model = model::Model::load(
            &config.model,
            config.main_gpu,
            config.context_size as usize,
            config.batch_size as usize,
        )
        .map_err(map)?;
        if model.cfg.vocab != tokenizer.n_vocab() {
            return Err(Error::UnsupportedModel);
        }
        let mut model = model;
        model.warmup().map_err(map)?;
        let images = match &config.mmproj {
            None => None,
            Some(path) => {
                let encoder =
                    Vision::load(model.gpu(), path, model.cfg.d, MAX_IMAGE_TOKENS).map_err(map)?;
                let sample = encoder.warmup().map_err(map)?;
                let warm = [
                    Segment::Tokens(&[2]),
                    Segment::Image {
                        rows: &sample.rows,
                        count: sample.tokens,
                        key: 0,
                    },
                ];
                model.prefill_segments(&warm).map_err(map)?;
                model.clear_prompt_cache();
                let delimiter = |text: &str| match tokenizer.tokenize(text, false, true)[..] {
                    [token] => Ok(token),
                    _ => Err(Error::UnsupportedModel),
                };
                Some(Images {
                    encoder,
                    begin: delimiter("<|image>")?,
                    end: delimiter("<image|>")?,
                    encoded: Vec::new(),
                })
            }
        };
        Ok(Self {
            images,
            n_ctx: config.context_size as usize,
            batch_size: config.batch_size as usize,
            prompt_cache: config.prompt_cache,
            model,
            tokenizer,
            profile: PrefillProfile::default(),
        })
    }
}

impl Backend for CubeBackend {
    fn n_vocab(&self) -> i32 {
        self.tokenizer.n_vocab() as i32
    }

    fn mask(&self) -> i32 {
        self.tokenizer.mask()
    }

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        Ok(self.tokenizer.tokenize(text, add_special, parse_special))
    }

    fn code_piece(&self, token: i32) -> Option<String> {
        if token < 0
            || token as usize >= self.tokenizer.n_vocab()
            || self.tokenizer.is_control(token)
        {
            return None;
        }
        let bytes = self.tokenizer.token_to_piece(token);
        if !(1..=16).contains(&bytes.len()) || !bytes.iter().all(u8::is_ascii_alphanumeric) {
            return None;
        }
        String::from_utf8(bytes).ok()
    }

    fn image_parts(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let state = self.images.as_mut().ok_or_else(|| {
            Error::InvalidInput(
                "Image input requires --mmproj with a compatible DiffusionGemma vision projector"
                    .into(),
            )
        })?;
        if images.len() > 8 {
            return Err(Error::InvalidInput("At most 8 images are allowed".into()));
        }
        state.encoded.clear();
        let mut parts = Vec::with_capacity(3 * images.len());
        for (index, image) in images.iter().enumerate() {
            let rgb = crate::images::decode(image)?;
            let rgb = Rgb {
                width: rgb.width() as usize,
                height: rgb.height() as usize,
                data: rgb.into_raw(),
            };
            let tokens = state.encoder.tokens_for(rgb.width, rgb.height);
            if tokens > self.batch_size {
                return Err(Error::InvalidInput(format!(
                    "Image needs {tokens} tokens in one batch; increase --batch-size"
                )));
            }
            let mut hasher = DefaultHasher::new();
            (rgb.width, rgb.height, &rgb.data).hash(&mut hasher);
            let encoded = state.encoder.encode(&rgb).map_err(map)?;
            parts.push(PromptPart::Text(vec![state.begin]));
            parts.push(PromptPart::Image {
                tokens: encoded.tokens,
                index,
            });
            parts.push(PromptPart::Text(vec![state.end]));
            state.encoded.push((encoded, hasher.finish()));
        }
        Ok(parts)
    }

    fn prefill(&mut self, parts: &[PromptPart], suffix: &[i32]) -> Result<usize> {
        let start = Instant::now();
        let length = parts.iter().map(PromptPart::len).sum::<usize>() + suffix.len();
        if length > self.n_ctx {
            return Err(Error::InvalidInput("Prompt exceeds context size".into()));
        }
        let encoded = self.images.as_ref().map_or(&[][..], |s| &s.encoded[..]);
        let mut segments = Vec::with_capacity(parts.len() + 1);
        for part in parts {
            segments.push(match part {
                PromptPart::Text(tokens) => Segment::Tokens(tokens),
                PromptPart::Image { tokens, index } => {
                    let (image, key) = encoded
                        .get(*index)
                        .filter(|(image, _)| image.tokens == *tokens)
                        .ok_or_else(|| Error::InvalidInput("Unknown image part".into()))?;
                    Segment::Image {
                        rows: &image.rows,
                        count: *tokens,
                        key: *key,
                    }
                }
            });
        }
        segments.push(Segment::Tokens(suffix));
        if !self.prompt_cache {
            self.model.clear_prompt_cache();
        }
        let stats = self.model.prefill_segments(&segments).map_err(map)?;
        self.profile.wall_ms += start.elapsed().as_secs_f64() * 1000.0;
        self.profile.calls += 1;
        self.profile.batches += stats.batches;
        self.profile.processed_tokens += stats.processed_tokens;
        self.profile.reused_tokens += stats.reused_tokens;
        Ok(length)
    }

    fn decode_canvas(
        &mut self,
        tokens: &[i32],
        prompt_length: usize,
        previous: Option<&[f32]>,
        inverse_temperature: f32,
        full_logits: bool,
    ) -> Result<()> {
        self.model
            .canvas(
                tokens,
                prompt_length,
                full_logits,
                previous.map(|p| (p, inverse_temperature)),
            )
            .map_err(map)
    }

    fn all_logits(&mut self) -> Result<Vec<f32>> {
        self.model.all_logits().map_err(|_| Error::MissingLogits)
    }

    fn logits(&mut self, position: usize, candidates: &[i32]) -> Result<Vec<f64>> {
        self.model
            .candidate_logits(position, candidates)
            .map_err(map)
    }

    fn profile(&mut self) -> &mut PrefillProfile {
        &mut self.profile
    }
}
