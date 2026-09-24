//! DiffusionGemma on the CubeCL/HIP runtime: Rust tokenizer, GPU model, vision encoder and
//! prompt-prefix reuse.
use jevons_core::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Error, ImageInput, Logits,
    ModelConfig, ModelInfo, PrefillProfile, PromptPart, Result, TextTokenizer, decode_image,
};
use jevons_cubecl::model::Segment;
use jevons_cubecl::vision::{EncodedImage, Vision};
use jevons_cubecl::{gguf::Gguf, model, tokenizer::Tokenizer, vision_input::Rgb};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

/// Image tokens per image, as in llama.cpp's DiffusionGemma integration.
const MAX_IMAGE_TOKENS: usize = 280;
/// Largest canvas per forward, matching the pinned native sampler.
const MAX_CANVAS: usize = 64;

struct Images {
    encoder: Vision,
    /// `<|image>` and `<image|>` delimiter tokens.
    begin: i32,
    end: i32,
    /// Images encoded for the current request, with content keys for prompt reuse.
    encoded: Vec<(EncodedImage, u64)>,
}

struct GemmaTokenizer(Tokenizer);

impl TextTokenizer for GemmaTokenizer {
    fn tokenize(&self, text: &str, bos: bool, special: bool) -> Result<Vec<i32>> {
        Ok(self.0.tokenize(text, bos, special))
    }

    fn code_piece(&self, token: i32) -> Option<String> {
        if token < 0 || token as usize >= self.0.n_vocab() || self.0.is_control(token) {
            return None;
        }
        let bytes = self.0.token_to_piece(token);
        if !(1..=16).contains(&bytes.len()) || !bytes.iter().all(u8::is_ascii_alphanumeric) {
            return None;
        }
        String::from_utf8(bytes).ok()
    }
}

pub(crate) struct Gemma4 {
    model: model::Model,
    tokenizer: GemmaTokenizer,
    info: ModelInfo,
    chat: ChatFormat,
    profile: PrefillProfile,
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

fn chat_format() -> ChatFormat {
    ChatFormat {
        bos: true,
        user_open: "<|turn>user\n".into(),
        model_open: "<turn|>\n<|turn>model\n".into(),
        thought_open: "<|channel>thought\n".into(),
        thought_close: "<channel|>".into(),
        empty_thought: "<|channel>thought\n<channel|>".into(),
        thought_stops: vec!["<channel|>".into(), "<turn|>".into()],
        space_joins_answers: false,
    }
}

impl Gemma4 {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let gguf = Gguf::open(&config.model).map_err(|_| Error::ModelLoad)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)
            .map_err(|e| Error::UnsupportedModel(format!("DiffusionGemma tokenizer: {e}")))?;
        drop(gguf);
        let model = model::Model::load(
            &config.model,
            config.main_gpu,
            config.context_size as usize,
            config.batch_size as usize,
        )
        .map_err(map)?;
        if model.cfg.vocab != tokenizer.n_vocab() {
            return Err(Error::UnsupportedModel(
                "tokenizer and model vocabulary sizes differ".into(),
            ));
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
                    Segment::Tokens(&[tokenizer.bos()]),
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
                    _ => Err(Error::UnsupportedModel(format!(
                        "the tokenizer has no single token for the image marker {text}"
                    ))),
                };
                Some(Images {
                    encoder,
                    begin: delimiter("<|image>")?,
                    end: delimiter("<image|>")?,
                    encoded: Vec::new(),
                })
            }
        };
        let info = ModelInfo {
            architecture: "gemma4-diffusion",
            display_name: "DiffusionGemma GGUF".into(),
            n_vocab: tokenizer.n_vocab() as i32,
            n_ctx: config.context_size as usize,
            batch_size: config.batch_size as usize,
            max_canvas: MAX_CANVAS,
        };
        Ok(Self {
            images,
            info,
            chat: chat_format(),
            prompt_cache: config.prompt_cache,
            model,
            tokenizer: GemmaTokenizer(tokenizer),
            profile: PrefillProfile::default(),
        })
    }
}

impl DiffusionModel for Gemma4 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenizer(&self) -> &dyn TextTokenizer {
        &self.tokenizer
    }

    fn chat(&self) -> &ChatFormat {
        &self.chat
    }

    fn scheme(&self) -> DiffusionScheme {
        DiffusionScheme::UniformSelfConditioned {
            mask: self.tokenizer.0.mask(),
        }
    }

    fn encode_images(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
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
            let image = decode_image(image)?;
            let rgb = Rgb {
                width: image.width,
                height: image.height,
                data: image.data,
            };
            let tokens = state.encoder.tokens_for(rgb.width, rgb.height);
            if tokens > self.info.batch_size {
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
        if length > self.info.n_ctx {
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

    fn forward_canvas(
        &mut self,
        tokens: &[i32],
        prompt_length: usize,
        conditioning: Conditioning<'_>,
        logits: Logits,
    ) -> Result<()> {
        let previous = match conditioning {
            Conditioning::None => None,
            Conditioning::Previous {
                logits,
                inverse_temperature,
            } => Some((logits, inverse_temperature)),
        };
        self.model
            .canvas(tokens, prompt_length, logits == Logits::Full, previous)
            .map_err(map)
    }

    fn candidate_logits(&mut self, row: usize, candidates: &[i32]) -> Result<Vec<f64>> {
        self.model.candidate_logits(row, candidates).map_err(map)
    }

    fn full_logits(&mut self) -> Result<Vec<f32>> {
        self.model.all_logits().map_err(|_| Error::MissingLogits)
    }

    fn profile(&mut self) -> &mut PrefillProfile {
        &mut self.profile
    }
}
