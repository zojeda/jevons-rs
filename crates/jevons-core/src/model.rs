//! The contract between the engine and a diffusion model architecture.
//!
//! A model owns its weights, tokenizer, prompt KV state and device buffers on the inference
//! worker thread; nothing here requires `Send`. The engine drives reads through this trait and
//! keeps sampling policy to itself, selected by the model's [`DiffusionScheme`].
use crate::{Error, ImageInput, PrefillProfile, Result};

pub enum PromptPart {
    Text(Vec<i32>),
    /// An image encoded by [`DiffusionModel::encode_images`]; `index` names it within that call.
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Static facts about a loaded model.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// Stable architecture identifier, such as `gemma4-diffusion`.
    pub architecture: &'static str,
    /// Human-readable model description for logs and model listings.
    pub display_name: String,
    pub n_vocab: i32,
    /// Maximum prompt + canvas positions.
    pub n_ctx: usize,
    /// Maximum tokens evaluated in one forward.
    pub batch_size: usize,
    /// Largest canvas the engine should evaluate in one forward.
    pub max_canvas: usize,
}

/// Chat framing around a read. Every marker is parsed with special tokens enabled; user text
/// never is.
#[derive(Clone, Debug)]
pub struct ChatFormat {
    /// Prepend the tokenizer's beginning-of-sequence token to the prompt.
    pub bos: bool,
    /// Opens the user turn, before any images and the prompt text.
    pub user_open: String,
    /// Closes the user turn and opens the model turn.
    pub model_open: String,
    /// Opens a thought for bounded thinking.
    pub thought_open: String,
    /// Closes a thought; appended after generated thought tokens.
    pub thought_close: String,
    /// Suffix that states an empty thought when thinking is off.
    pub empty_thought: String,
    /// Single-token markers that end a generated thought.
    pub thought_stops: Vec<String>,
    /// The tokenizer joins a leading space to the following word (byte-level BPE). A slot
    /// prefix's trailing space then moves onto its candidates, so `"Answer: " + "A"` is read
    /// as `"Answer:" + " A"`, the tokens the model saw in training.
    pub space_joins_answers: bool,
}

impl ChatFormat {
    /// Checks that each stop marker is one token, so generation can stop on it.
    pub fn validate(&self, tokenizer: &dyn TextTokenizer) -> Result<()> {
        for marker in &self.thought_stops {
            if tokenizer.tokenize(marker, false, true)?.len() != 1 {
                return Err(Error::UnsupportedModel(format!(
                    "the tokenizer has no single token for the chat marker {marker}"
                )));
            }
        }
        Ok(())
    }
}

/// How the model was trained to denoise, which selects the engine's sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiffusionScheme {
    /// Canvas positions start as uniform random tokens; later steps condition on the previous
    /// step's full logits. `mask` is excluded from noise.
    UniformSelfConditioned { mask: i32 },
    /// Canvas positions start as `mask` and are committed by confidence. `block` is the
    /// generation block size and `threshold` the commit probability.
    Masked {
        mask: i32,
        block: usize,
        threshold: f64,
        max_steps: usize,
    },
}

/// Extra input to a canvas forward.
#[derive(Clone, Copy, Debug)]
pub enum Conditioning<'a> {
    None,
    /// The prior step's full logits (canvas rows x vocab) at an inverse temperature.
    Previous {
        logits: &'a [f32],
        inverse_temperature: f32,
    },
}

/// Which logits a canvas forward must make available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logits {
    /// Only [`DiffusionModel::candidate_logits`] is used.
    Candidates,
    /// [`DiffusionModel::full_logits`] is used as well.
    Full,
}

pub trait TextTokenizer {
    /// Encodes `text`. `bos` prepends the beginning-of-sequence token; `special` parses control
    /// markers. With `special = false` text never produces control or added special tokens.
    fn tokenize(&self, text: &str, bos: bool, special: bool) -> Result<Vec<i32>>;
    /// Decoded text of a non-control token if it is 1..=16 ASCII alphanumerics, ignoring one
    /// leading space.
    fn code_piece(&self, token: i32) -> Option<String>;
}

pub trait DiffusionModel {
    fn info(&self) -> &ModelInfo;
    fn tokenizer(&self) -> &dyn TextTokenizer;
    fn chat(&self) -> &ChatFormat;
    fn scheme(&self) -> DiffusionScheme;
    /// Decodes (with [`crate::decode_image`]) and encodes images and returns their prompt parts,
    /// delimiters included. The parts stay valid until the next call.
    fn encode_images(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>>;
    /// Makes `parts + suffix` the resident causal prompt, reusing any cached common prefix, and
    /// returns its length. Appending generated tokens to `suffix` commits them to the cache.
    fn prefill(&mut self, parts: &[PromptPart], suffix: &[i32]) -> Result<usize>;
    /// Bidirectional canvas forward over the resident prompt; the prompt cache is unchanged.
    fn forward_canvas(
        &mut self,
        tokens: &[i32],
        prompt_length: usize,
        conditioning: Conditioning<'_>,
        logits: Logits,
    ) -> Result<()>;
    /// Logits of `candidates` at canvas row `row` of the last canvas forward.
    fn candidate_logits(&mut self, row: usize, candidates: &[i32]) -> Result<Vec<f64>>;
    /// All logits (canvas rows x vocab) of the last canvas forward with [`Logits::Full`].
    fn full_logits(&mut self) -> Result<Vec<f32>>;
    /// Prefill work counters, reset by the engine at the start of each read.
    fn profile(&mut self) -> &mut PrefillProfile;
}
