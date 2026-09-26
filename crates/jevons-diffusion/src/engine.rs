//! The diffusion engine shared by the Generative and Decision services: the loaded model, its
//! chat framing and answer codes, context limits, and bounded token generation (thoughts and
//! answers) with every decoding mode.

use crate::sampler::{masked, uniform};
use jevons_core::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Error, ImageInput, Logits,
    PrefillProfile, PromptPart, Result, TextTokenizer,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::{collections::HashSet, time::Instant};

/// How text (thoughts and answers) is generated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Decoding {
    /// The model's diffusion sampler. For masked models, each block starts with the causal
    /// prediction after the committed text and its masks are unmasked by confidence (the
    /// reference `generate`).
    #[default]
    Diffusion,
    /// Linear self-speculation (the reference `linear_spec_generate`): one bidirectional
    /// forward drafts a block, one causal forward verifies it, and the longest prefix the
    /// causal predictions agree with is kept, plus the prediction after it. The tokens equal
    /// greedy autoregressive decoding.
    SelfSpeculation,
    /// Greedy autoregressive decoding, one causal forward per token.
    Autoregressive,
}

impl Decoding {
    pub const ALL: [Self; 3] = [Self::Diffusion, Self::SelfSpeculation, Self::Autoregressive];

    /// The `--decoding` value.
    pub fn id(self) -> &'static str {
        match self {
            Self::Diffusion => "diffusion",
            Self::SelfSpeculation => "self-speculation",
            Self::Autoregressive => "autoregressive",
        }
    }
}

impl std::str::FromStr for Decoding {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|d| d.id() == name)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Unknown decoding {name:?}; expected diffusion, self-speculation or autoregressive"
                ))
            })
    }
}

/// A loaded diffusion model with everything both services need to drive it.
pub struct DiffusionEngine {
    model: Box<dyn DiffusionModel>,
    chat: ChatFormat,
    codes: Vec<String>,
    n_vocab: i32,
    scheme: DiffusionScheme,
    /// Token excluded from answer codes and noise.
    mask: i32,
    n_ctx: usize,
    batch_size: usize,
    max_canvas: usize,
    decoding: Decoding,
}

impl DiffusionEngine {
    /// Detects and loads the model named by `config`, then prepares the engine.
    #[cfg(feature = "models")]
    pub fn load(config: &jevons_core::ModelConfig) -> Result<Self> {
        Self::new(jevons_models::load(config)?)
    }

    pub fn new(model: Box<dyn DiffusionModel>) -> Result<Self> {
        let chat = model.chat().clone();
        chat.validate(model.tokenizer())?;
        let scheme = model.scheme();
        let mask = match scheme {
            DiffusionScheme::UniformSelfConditioned { mask }
            | DiffusionScheme::Masked { mask, .. } => mask,
        };
        let info = model.info().clone();
        let mut engine = Self {
            n_vocab: info.n_vocab,
            scheme,
            mask,
            n_ctx: info.n_ctx,
            batch_size: info.batch_size,
            max_canvas: info.max_canvas,
            decoding: Decoding::Diffusion,
            chat,
            model,
            codes: Vec::new(),
        };
        engine.codes = engine.find_codes(128)?;
        Ok(engine)
    }

    /// Selects how thoughts are generated. Self-speculation and autoregressive decoding need a
    /// masked diffusion model with causal next-token predictions.
    pub fn set_decoding(&mut self, decoding: Decoding) -> Result<()> {
        if decoding != Decoding::Diffusion && !matches!(self.scheme, DiffusionScheme::Masked { .. })
        {
            return Err(Error::InvalidInput(format!(
                "Decoding {} needs a masked diffusion model with causal predictions",
                decoding.id()
            )));
        }
        self.decoding = decoding;
        Ok(())
    }

    pub fn decoding(&self) -> Decoding {
        self.decoding
    }

    /// Verified single-token codes used to represent arbitrary external labels.
    pub fn codes(&self) -> &[String] {
        &self.codes
    }

    /// Architecture and description of the loaded model.
    pub fn model_info(&self) -> &jevons_core::ModelInfo {
        self.model.info()
    }

    /// Synchronized prefill work for the last read; independent of logical usage.
    pub fn prefill_profile(&mut self) -> PrefillProfile {
        *self.model.profile()
    }

    pub fn tokenize(&self, text: &str, bos: bool, special: bool) -> Result<Vec<i32>> {
        self.model.tokenizer().tokenize(text, bos, special)
    }

    /// Candidate text as it is tokenized in an answer slot.
    pub fn answer_text(&self, candidate: &str) -> String {
        if self.chat.space_joins_answers {
            format!(" {candidate}")
        } else {
            candidate.to_string()
        }
    }

    /// Slot prefix tokens; a trailing space moves onto the candidates when spaces join words.
    pub fn prefix_tokens(&self, prefix: &str) -> Result<Vec<i32>> {
        let prefix = match prefix.strip_suffix(' ') {
            Some(stripped) if self.chat.space_joins_answers => stripped,
            _ => prefix,
        };
        self.tokenize(prefix, false, false)
    }

    /// Canvas length for `content` tokens: masked canvases are padded with masks to a whole
    /// block, within the canvas capacity.
    pub fn padded(&self, content: usize) -> usize {
        match self.scheme {
            DiffusionScheme::Masked { block, .. } => content
                .div_ceil(block)
                .saturating_mul(block)
                .min(self.max_canvas.min(self.batch_size))
                .max(content),
            DiffusionScheme::UniformSelfConditioned { .. } => content,
        }
    }

    fn find_codes(&self, count: usize) -> Result<Vec<String>> {
        let mut codes = Vec::with_capacity(count);
        let mut seen = HashSet::new();
        for ch in ('A'..='Z').chain('a'..='z').chain('0'..='9') {
            let code = ch.to_string();
            let tokens = self.tokenize(&self.answer_text(&code), false, false)?;
            if tokens.len() == 1 && seen.insert(tokens[0]) {
                codes.push(code);
            }
        }
        for token in 0..self.n_vocab {
            if codes.len() >= count {
                break;
            }
            if seen.contains(&token) || token == self.mask {
                continue;
            }
            if let Some(code) = self.model.tokenizer().code_piece(token)
                && self.tokenize(&self.answer_text(&code), false, false)? == [token]
            {
                seen.insert(token);
                codes.push(code);
            }
        }
        if codes.len() < count {
            return Err(Error::InvalidInput(format!(
                "The vocabulary has fewer than {count} usable single-token answer codes"
            )));
        }
        Ok(codes)
    }

    /// The loaded model, for services that drive it directly.
    pub fn model(&mut self) -> &mut dyn DiffusionModel {
        self.model.as_mut()
    }

    pub fn tokenizer(&self) -> &dyn TextTokenizer {
        self.model.tokenizer()
    }

    /// Chat markers and answer framing of the loaded model.
    pub fn chat(&self) -> &ChatFormat {
        &self.chat
    }

    pub fn scheme(&self) -> DiffusionScheme {
        self.scheme
    }

    /// The model's mask token, excluded from answer codes and noise.
    pub fn mask(&self) -> i32 {
        self.mask
    }

    pub fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    /// Context length in tokens: prompt, thought and canvas or answer must fit it.
    pub fn context_size(&self) -> usize {
        self.n_ctx
    }

    /// Lowers (or restores) the usable context, such as to test exact limits.
    pub fn set_context_size(&mut self, tokens: usize) {
        self.n_ctx = tokens;
    }

    /// The largest canvas one forward accepts.
    pub fn canvas_capacity(&self) -> usize {
        self.max_canvas.min(self.batch_size)
    }

    /// Clears the prefill diagnostics before a new request.
    pub fn reset_profile(&mut self) {
        *self.model.profile() = PrefillProfile::default();
    }

    /// The user turn with any images, then the opened model turn.
    pub fn prompt_parts(&mut self, text: &str, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
        let mut prompt = vec![PromptPart::Text(self.tokenize(
            &self.chat.user_open,
            self.chat.bos,
            true,
        )?)];
        prompt.extend(self.model.encode_images(images)?);
        prompt.push(PromptPart::Text(self.tokenize(
            text.trim(),
            false,
            false,
        )?));
        prompt.push(PromptPart::Text(self.tokenize(
            &self.chat.model_open,
            false,
            true,
        )?));
        Ok(prompt)
    }

    /// The thought's opening tokens, closing tokens and single-token stop markers.
    pub fn thought_markers(&self) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let open = self.tokenize(&self.chat.thought_open, false, true)?;
        let close = self.tokenize(&self.chat.thought_close, false, true)?;
        let stops = self.marker_tokens(&self.chat.thought_stops)?;
        Ok((open, close, stops))
    }

    pub fn marker_tokens(&self, markers: &[String]) -> Result<Vec<i32>> {
        markers.iter().map(|m| self.single_token(m)).collect()
    }

    pub fn single_token(&self, marker: &str) -> Result<i32> {
        match self.tokenize(marker, false, true)?[..] {
            [token] => Ok(token),
            _ => Err(Error::UnsupportedModel(format!(
                "the tokenizer has no single token for the chat marker {marker}"
            ))),
        }
    }

    /// A bounded thought after `prompt`, generated with `decoding`.
    pub fn think_with(
        &mut self,
        prompt: &[PromptPart],
        budget: usize,
        seed: u64,
        decoding: Decoding,
    ) -> Result<Thought> {
        let (open, close, stops) = self.thought_markers()?;
        let generated = self.generate_tokens(
            prompt,
            &open,
            budget,
            &stops,
            seed,
            decoding,
            &mut |_, _| true,
        )?;
        Ok(Thought {
            output_tokens: generated.tokens.len() + usize::from(generated.stopped),
            suffix: [open, generated.tokens, close].concat(),
            input_tokens: generated.input_tokens,
            forward_ms: generated.forward_ms,
            forwards: generated.forwards,
            accepted: generated.accepted,
        })
    }

    /// Generates up to `budget` tokens after `prompt + start`, ending before the first of
    /// `stops`. `sink` receives each newly decided run of tokens and returns false to end early.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_tokens(
        &mut self,
        prompt: &[PromptPart],
        start: &[i32],
        budget: usize,
        stops: &[i32],
        seed: u64,
        decoding: Decoding,
        sink: Sink<'_>,
    ) -> Result<Generated> {
        match self.scheme {
            DiffusionScheme::UniformSelfConditioned { .. } => {
                self.generate_uniform(prompt, start, budget, stops, seed, sink)
            }
            DiffusionScheme::Masked {
                block,
                threshold,
                max_steps,
                ..
            } => match decoding {
                Decoding::Diffusion => self.generate_masked(
                    prompt, start, budget, stops, block, threshold, max_steps, sink,
                ),
                Decoding::SelfSpeculation => {
                    self.generate_causal(prompt, start, budget, stops, block, sink)
                }
                Decoding::Autoregressive => {
                    self.generate_causal(prompt, start, budget, stops, 1, sink)
                }
            },
        }
    }

    /// Masked block diffusion (the reference `generate` with causal context). Each block starts
    /// with the causal prediction after the committed text; its masks are unmasked by
    /// full-vocabulary confidence and the block is truncated at the first stop marker. The next
    /// block's prefill commits it to the prompt cache and predicts that block's first token.
    #[allow(clippy::too_many_arguments)]
    fn generate_masked(
        &mut self,
        prompt: &[PromptPart],
        start: &[i32],
        budget: usize,
        stops: &[i32],
        block: usize,
        threshold: f64,
        max_steps: usize,
        sink: Sink<'_>,
    ) -> Result<Generated> {
        let mut suffix = start.to_vec();
        let mut generated = Generated::default();
        while generated.tokens.len() < budget {
            let (prompt_length, first) = self.model.prefill_predict(prompt, &suffix, 1)?;
            generated.input_tokens += prompt_length;
            generated.forwards += 1;
            let count = block
                .min(self.max_canvas)
                .min(self.batch_size)
                .min(budget - generated.tokens.len());
            let mut canvas = vec![self.mask; count];
            canvas[0] = *first.first().ok_or(Error::InvalidLogits)?;
            let mut open = vec![true; count];
            open[0] = false;
            let clock = Instant::now();
            for step in 0..max_steps.max(1) {
                // Stop once a stop marker is fixed and everything before it is too.
                let settled = open.iter().position(|&o| o).unwrap_or(count);
                if settled == count || canvas[..settled].iter().any(|t| stops.contains(t)) {
                    break;
                }
                self.model.forward_canvas(
                    &canvas,
                    prompt_length,
                    Conditioning::None,
                    Logits::Full,
                )?;
                generated.forwards += 1;
                let proposals = self.model.greedy_proposals()?;
                if proposals.len() != count {
                    return Err(Error::InvalidLogits);
                }
                let rows: Vec<usize> = (0..count).filter(|&i| open[i]).collect();
                let proposals: Vec<_> = rows
                    .iter()
                    .map(|&i| masked::Proposal {
                        token: proposals[i].0,
                        confidence: proposals[i].1,
                    })
                    .collect();
                let committed = if step + 1 == max_steps.max(1) {
                    (0..rows.len()).collect()
                } else {
                    masked::commits(&proposals, threshold)
                };
                for k in committed {
                    canvas[rows[k]] = proposals[k].token;
                    open[rows[k]] = false;
                }
            }
            generated.forward_ms += clock.elapsed().as_secs_f64() * 1000.0;
            let settled = open.iter().position(|&o| o).unwrap_or(count);
            let stop = canvas[..settled].iter().position(|t| stops.contains(t));
            let length = stop.unwrap_or(settled);
            if !generated.push(
                &canvas[..length],
                stop.is_some(),
                self.model.tokenizer(),
                sink,
            ) || length == 0
            {
                break;
            }
            suffix.extend(&canvas[..length]);
        }
        Ok(generated)
    }

    /// Greedy causal decoding, verifying up to `block` tokens per causal forward.
    ///
    /// The last decided token is pending: predicted, but not yet in the prompt cache. With
    /// `block > 1` this is linear self-speculation: one bidirectional forward fills the masks
    /// after the pending token, one causal forward over the block predicts each next token,
    /// and the drafts are kept while they match those predictions, followed by the prediction
    /// after the last match. Cached rows past the kept tokens are dropped by the next prefill.
    /// With `block == 1` it is plain autoregressive decoding; both give the same tokens.
    fn generate_causal(
        &mut self,
        prompt: &[PromptPart],
        start: &[i32],
        budget: usize,
        stops: &[i32],
        block: usize,
        sink: Sink<'_>,
    ) -> Result<Generated> {
        let mut generated = Generated::default();
        if budget == 0 {
            return Ok(generated);
        }
        let (prompt_length, first) = self.model.prefill_predict(prompt, start, 1)?;
        generated.input_tokens += prompt_length;
        generated.forwards += 1;
        let first = *first.first().ok_or(Error::InvalidLogits)?;
        let stopped = stops.contains(&first);
        let tokenizer = self.model.tokenizer();
        let mut going = generated.push(&[first][..usize::from(!stopped)], stopped, tokenizer, sink);
        let capacity = block.max(1).min(self.max_canvas).min(self.batch_size);
        // The decided tokens are `generated.tokens`; the last one is pending.
        while going && generated.tokens.len() < budget {
            let text = &generated.tokens;
            let count = capacity.min(budget - text.len());
            let pending = text[text.len() - 1];
            let mut resident = [start, &text[..text.len() - 1]].concat();
            let mut drafts = Vec::new();
            if count > 1 {
                let prompt_length = self.model.prefill(prompt, &resident)?;
                generated.input_tokens += prompt_length;
                let mut canvas = vec![self.mask; count];
                canvas[0] = pending;
                let clock = Instant::now();
                self.model.forward_canvas(
                    &canvas,
                    prompt_length,
                    Conditioning::None,
                    Logits::Full,
                )?;
                let proposals = self.model.greedy_proposals()?;
                generated.forward_ms += clock.elapsed().as_secs_f64() * 1000.0;
                generated.forwards += 1;
                if proposals.len() != count {
                    return Err(Error::InvalidLogits);
                }
                drafts.extend(proposals[1..].iter().map(|&(token, _)| token));
            }
            resident.push(pending);
            resident.extend(&drafts);
            let (prompt_length, predicted) =
                self.model
                    .prefill_predict(prompt, &resident, drafts.len() + 1)?;
            if count == 1 {
                generated.input_tokens += prompt_length;
            }
            generated.forwards += 1;
            if predicted.len() != drafts.len() + 1 {
                return Err(Error::InvalidLogits);
            }
            let kept = drafts
                .iter()
                .zip(&predicted)
                .take_while(|(draft, prediction)| draft == prediction)
                .count();
            let decided: Vec<i32> = drafts[..kept]
                .iter()
                .chain([&predicted[kept]])
                .copied()
                .collect();
            let stop = decided.iter().position(|t| stops.contains(t));
            let length = stop.unwrap_or(decided.len());
            generated.accepted.push(kept + 1);
            going = generated.push(
                &decided[..length],
                stop.is_some(),
                self.model.tokenizer(),
                sink,
            );
        }
        Ok(generated)
    }

    /// Uniform-noise diffusion with self-conditioning (DiffusionGemma): blocks of up to 64
    /// tokens refined by the pinned entropy-bound denoiser, at most 48 iterations each.
    fn generate_uniform(
        &mut self,
        prompt: &[PromptPart],
        start: &[i32],
        budget: usize,
        stops: &[i32],
        seed: u64,
        sink: Sink<'_>,
    ) -> Result<Generated> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut suffix = start.to_vec();
        let mut generated = Generated::default();
        while generated.tokens.len() < budget {
            let prompt_length = self.model.prefill(prompt, &suffix)?;
            generated.input_tokens += prompt_length;
            let count = self
                .max_canvas
                .min(self.batch_size)
                .min(budget - generated.tokens.len());
            let mut canvas: Vec<_> = (0..count)
                .map(|_| uniform::noise(self.n_vocab, self.mask, &mut rng))
                .collect();
            let positions: Vec<_> = (0..count).collect();
            let mut previous: Option<Vec<f32>> = None;
            let mut previous_best = Vec::new();
            let mut inverse_temperature = 1.0;
            let clock = Instant::now();
            for step in 0..48 {
                self.model.forward_canvas(
                    &canvas,
                    prompt_length,
                    conditioning(previous.as_deref(), inverse_temperature),
                    Logits::Full,
                )?;
                generated.forwards += 1;
                let logits = self.model.full_logits()?;
                let temperature = 0.4 + 0.4 * (48 - step) as f64 / 48.0;
                let predictions = uniform::refine(
                    &mut canvas,
                    &logits,
                    &positions,
                    self.n_vocab as usize,
                    temperature,
                    self.mask,
                    &mut rng,
                )?;
                let best: Vec<_> = predictions.iter().map(|p| p.best).collect();
                let stable = best == previous_best
                    && predictions.iter().map(|p| p.entropy).sum::<f64>() / (count as f64) < 0.005;
                previous_best = best;
                if stable {
                    break;
                }
                previous = Some(logits);
                inverse_temperature = (1.0 / temperature) as f32;
            }
            generated.forward_ms += clock.elapsed().as_secs_f64() * 1000.0;
            let stop = previous_best.iter().position(|t| stops.contains(t));
            let length = stop.unwrap_or(previous_best.len());
            if !generated.push(
                &previous_best[..length],
                stop.is_some(),
                self.model.tokenizer(),
                sink,
            ) {
                break;
            }
            suffix.extend(&previous_best[..length]);
        }
        Ok(generated)
    }
}

pub fn conditioning(previous: Option<&[f32]>, inverse_temperature: f32) -> Conditioning<'_> {
    match previous {
        Some(logits) => Conditioning::Previous {
            logits,
            inverse_temperature,
        },
        None => Conditioning::None,
    }
}

/// Receives each newly decided run of generated tokens; returns false to end generation.
pub type Sink<'a> = &'a mut dyn FnMut(&[i32], &dyn TextTokenizer) -> bool;

/// Tokens generated by one decoding run, before any stop marker.
#[derive(Default)]
pub struct Generated {
    pub tokens: Vec<i32>,
    /// A stop marker ended generation.
    pub stopped: bool,
    /// The sink ended generation.
    pub cancelled: bool,
    pub input_tokens: usize,
    pub forwards: usize,
    pub forward_ms: f64,
    /// Tokens decided by each causal verification.
    pub accepted: Vec<usize>,
}

impl Generated {
    /// Appends decided tokens and passes them on; false when generation should end.
    pub fn push(
        &mut self,
        tokens: &[i32],
        stopped: bool,
        tokenizer: &dyn TextTokenizer,
        sink: Sink<'_>,
    ) -> bool {
        self.tokens.extend(tokens);
        self.stopped = stopped;
        if !tokens.is_empty() && !sink(tokens, tokenizer) {
            self.cancelled = true;
        }
        !(stopped || self.cancelled)
    }
}

#[derive(Default)]
pub struct Thought {
    pub suffix: Vec<i32>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub forward_ms: f64,
    /// Generation forwards and tokens per causal verification, printed by the model tests.
    #[allow(dead_code)]
    pub forwards: usize,
    #[allow(dead_code)]
    pub accepted: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeModel, fake_engine, masked_model};

    #[test]
    fn causal_decoding_needs_a_masked_model() {
        for decoding in Decoding::ALL {
            assert_eq!(decoding.id().parse::<Decoding>().unwrap(), decoding);
        }
        assert!("speculative".parse::<Decoding>().is_err());
        let (mut engine, _) = fake_engine(FakeModel::new());
        assert!(engine.set_decoding(Decoding::Diffusion).is_ok());
        assert!(engine.set_decoding(Decoding::SelfSpeculation).is_err());
        let (mut engine, _) = fake_engine(masked_model());
        assert!(engine.set_decoding(Decoding::Autoregressive).is_ok());
    }
}
