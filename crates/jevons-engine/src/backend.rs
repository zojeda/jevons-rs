//! Inference backend contract used by the engine.
//!
//! Backends own their model, tokenizer, prompt KV state and device buffers on the worker
//! thread; nothing here requires `Send`.
use crate::{ImageInput, PrefillProfile, Result};

pub(crate) enum PromptPart {
    Text(Vec<i32>),
    /// An image encoded by [`Backend::image_parts`]; `index` names it within that call.
    Image {
        tokens: usize,
        index: usize,
    },
}

impl PromptPart {
    pub fn len(&self) -> usize {
        match self {
            Self::Text(tokens) => tokens.len(),
            Self::Image { tokens, .. } => *tokens,
        }
    }
}

pub(crate) trait Backend {
    fn n_vocab(&self) -> i32;
    fn mask(&self) -> i32;
    /// Maximum prompt + canvas positions.
    fn n_ctx(&self) -> usize;
    /// Maximum tokens evaluated in one forward.
    fn batch_size(&self) -> usize;
    fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>>;
    /// Decoded text of a non-control token if it is 1..=16 ASCII alphanumerics.
    fn code_piece(&self, token: i32) -> Option<String>;
    /// Encodes images and returns their prompt parts, delimiters included. The parts stay valid
    /// until the next call.
    fn image_parts(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>>;
    /// Makes `parts + suffix` the resident prompt; returns its length.
    fn prefill(&mut self, parts: &[PromptPart], suffix: &[i32]) -> Result<usize>;
    /// Canvas forward over the resident prompt. `previous` holds the prior step's logits for
    /// self-conditioning. When `full_logits` is false only [`Backend::logits`] is used.
    fn decode_canvas(
        &mut self,
        tokens: &[i32],
        prompt_length: usize,
        previous: Option<&[f32]>,
        inverse_temperature: f32,
        full_logits: bool,
    ) -> Result<()>;
    fn all_logits(&mut self) -> Result<Vec<f32>>;
    fn logits(&mut self, position: usize, candidates: &[i32]) -> Result<Vec<f64>>;
    fn profile(&mut self) -> &mut PrefillProfile;
}
