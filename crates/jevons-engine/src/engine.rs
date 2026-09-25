//! Token preparation, repeated reads, and bounded diffusion generation.

use crate::sampler::{masked, uniform};
use crate::{Error, ReadRequest, ReadResult, Result, SlotRead, restricted_softmax};
use crate::{ImageInput, ReadOptions};
use jevons_core::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Logits, PromptPart, TextTokenizer,
};
use jevons_core::{FinishReason, Generation, GenerationPrompt, GenerationRequest, Role};
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

pub struct Engine {
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

impl Engine {
    /// Detects and loads the model named by `config`, then prepares the engine.
    #[cfg(feature = "models")]
    pub fn load(config: &crate::ModelConfig) -> Result<Self> {
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
    pub fn prefill_profile(&mut self) -> crate::PrefillProfile {
        *self.model.profile()
    }

    fn tokenize(&self, text: &str, bos: bool, special: bool) -> Result<Vec<i32>> {
        self.model.tokenizer().tokenize(text, bos, special)
    }

    /// Candidate text as it is tokenized in an answer slot.
    fn answer_text(&self, candidate: &str) -> String {
        if self.chat.space_joins_answers {
            format!(" {candidate}")
        } else {
            candidate.to_string()
        }
    }

    /// Slot prefix tokens; a trailing space moves onto the candidates when spaces join words.
    fn prefix_tokens(&self, prefix: &str) -> Result<Vec<i32>> {
        let prefix = match prefix.strip_suffix(' ') {
            Some(stripped) if self.chat.space_joins_answers => stripped,
            _ => prefix,
        };
        self.tokenize(prefix, false, false)
    }

    /// Canvas length for `content` tokens: masked canvases are padded with masks to a whole
    /// block, within the canvas capacity.
    fn padded(&self, content: usize) -> usize {
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

    pub fn read(&mut self, request: &ReadRequest, seed: u64) -> Result<ReadResult> {
        self.read_with_options(request, seed, ReadOptions::default(), &[])
    }

    pub fn read_with_options(
        &mut self,
        request: &ReadRequest,
        seed: u64,
        options: ReadOptions,
        images: &[ImageInput],
    ) -> Result<ReadResult> {
        *self.model.profile() = crate::PrefillProfile::default();
        options.validate()?;
        if request.prompt.trim().is_empty() || request.slots.is_empty() {
            return Err(Error::InvalidInput(
                "A prompt and at least one slot are required".into(),
            ));
        }
        if !images.is_empty() && (options.think > 0 || options.sequential) {
            return Err(Error::InvalidInput(
                "Images cannot be combined with think or sequential".into(),
            ));
        }
        let prompt = self.prompt_parts(&request.prompt, images)?;
        let base_length: usize = prompt.iter().map(PromptPart::len).sum();
        let prepared = request
            .slots
            .iter()
            .map(|slot| {
                if slot.candidates.is_empty() {
                    return Err(Error::InvalidInput("Each slot needs a candidate".into()));
                }
                let mut candidates = Vec::new();
                for candidate in &slot.candidates {
                    let tokens = self.tokenize(&self.answer_text(candidate), false, false)?;
                    if tokens.len() != 1 || candidates.contains(&tokens[0]) {
                        return Err(Error::InvalidInput(format!(
                            "Candidate {candidate:?} must encode to one distinct token"
                        )));
                    }
                    candidates.push(tokens[0]);
                }
                Ok(PreparedSlot {
                    prefix: self.prefix_tokens(&slot.prefix)?,
                    candidates,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let capacity = self.max_canvas.min(self.batch_size);
        let lengths: Vec<_> = prepared.iter().map(|s| s.prefix.len() + 1).collect();
        let groups = chunk_ranges(&lengths, capacity)?;
        let group_lengths: Vec<usize> = groups
            .iter()
            .map(|r| lengths[r.clone()].iter().sum::<usize>())
            .collect();
        let padding = group_lengths
            .iter()
            .map(|&n| self.padded(n) - n)
            .max()
            .unwrap_or(0);
        let canvas_reserve = if options.sequential {
            lengths.iter().sum::<usize>() + padding
        } else {
            group_lengths
                .iter()
                .map(|&n| self.padded(n))
                .max()
                .unwrap_or(0)
        };
        // Close the thought before reading answers when no thought was requested.
        // Keep this in prefill so it does not change answer slots or canvas noise.
        let empty_thought = self.tokenize(&self.chat.empty_thought, false, true)?;
        let thought_reserve = if options.think > 0 {
            options.think + empty_thought.len()
        } else {
            empty_thought.len()
        };
        let mut suffix = if options.think == 0 {
            empty_thought
        } else {
            Vec::new()
        };
        if base_length + canvas_reserve + thought_reserve > self.n_ctx {
            return Err(Error::InvalidInput(format!(
                "Prompt, thought budget, and canvas need {} tokens; context allows {}",
                base_length + canvas_reserve + thought_reserve,
                self.n_ctx
            )));
        }
        let mut result = ReadResult {
            slots: Vec::new(),
            prompt_tokens: 0,
            canvas_tokens: 0,
            output_tokens: 0,
            seed,
            forward_ms: 0.0,
        };
        if options.think > 0 {
            let thought = self.think_with(&prompt, options.think, seed, self.decoding)?;
            suffix = thought.suffix;
            result.prompt_tokens += thought.input_tokens;
            result.output_tokens = thought.output_tokens;
            result.forward_ms += thought.forward_ms;
        }
        for (group_index, range) in groups.iter().enumerate() {
            let group = &prepared[range.clone()];
            let prompt_length = self.model.prefill(&prompt, &suffix)?;
            let group_seed = seed.wrapping_add(104729_u64.wrapping_mul(group_index as u64));
            let mut reads = Vec::new();
            for sample in 0..options.samples {
                let sample_seed = group_seed.wrapping_add(7919_u64.wrapping_mul(sample as u64));
                let (slots, canvas, time) = match self.scheme {
                    DiffusionScheme::UniformSelfConditioned { .. } => {
                        self.read_canvas(group, prompt_length, sample_seed, options.steps)?
                    }
                    DiffusionScheme::Masked { threshold, .. } => {
                        self.read_masked(group, prompt_length, options.steps, threshold)?
                    }
                };
                result.prompt_tokens += prompt_length;
                result.canvas_tokens += canvas;
                result.forward_ms += time;
                reads.push(slots);
            }
            let averaged = average_reads(reads)?;
            if options.sequential {
                for (slot, read) in group.iter().zip(&averaged) {
                    suffix.extend(&slot.prefix);
                    suffix.push(slot.candidates[best_index(&read.probabilities)]);
                }
            }
            result.slots.extend(averaged);
        }
        Ok(result)
    }

    /// Generates a free-form answer (the OpenAI-compatible endpoints). `on_text` receives the
    /// answer text in order as it is decided and returns false to stop, for example when the
    /// client has gone. Text that could still become a stop sequence is held back until it
    /// cannot.
    pub fn generate(
        &mut self,
        request: &GenerationRequest,
        seed: u64,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Generation> {
        request.validate()?;
        *self.model.profile() = crate::PrefillProfile::default();
        let (prompt, framing) = match &request.prompt {
            GenerationPrompt::Chat(messages) => {
                let mut tokens = if self.chat.bos {
                    self.tokenize("", true, false)?
                } else {
                    Vec::new()
                };
                for message in messages {
                    let (open, prefix) = match message.role {
                        Role::System => (&self.chat.system_open, ""),
                        Role::User => (&self.chat.user_open, ""),
                        Role::Assistant => {
                            (&self.chat.assistant_open, self.chat.history_prefix.as_str())
                        }
                    };
                    tokens.extend(self.tokenize(open, false, true)?);
                    tokens.extend(self.tokenize(prefix, false, true)?);
                    tokens.extend(self.tokenize(&message.text, false, false)?);
                    tokens.extend(self.tokenize(&self.chat.turn_close, false, true)?);
                }
                tokens.extend(self.tokenize(&self.chat.assistant_open, false, true)?);
                let framing = if request.think > 0 {
                    Vec::new()
                } else {
                    self.tokenize(&self.chat.empty_thought, false, true)?
                };
                (tokens, framing)
            }
            GenerationPrompt::Text(text) => {
                (self.tokenize(text, self.chat.bos, false)?, Vec::new())
            }
        };
        if prompt.is_empty() {
            return Err(Error::InvalidInput("The prompt is empty".into()));
        }
        let thought_reserve = if request.think > 0 {
            request.think + self.tokenize(&self.chat.thought_open, false, true)?.len() + 1
        } else {
            framing.len()
        };
        let available = self.n_ctx.saturating_sub(prompt.len() + thought_reserve);
        let max_tokens = request
            .max_tokens
            .unwrap_or(DEFAULT_MAX_TOKENS.min(available));
        if max_tokens == 0 || max_tokens > available {
            return Err(Error::InvalidInput(format!(
                "The prompt needs {} tokens and the answer up to {max_tokens}; the context allows {}",
                prompt.len() + thought_reserve,
                self.n_ctx
            )));
        }
        let prompt_tokens = prompt.len();
        let prompt = [PromptPart::Text(prompt)];
        let (start, reasoning_tokens) = if request.think > 0 {
            let thought = self.think_with(&prompt, request.think, seed, self.decoding)?;
            (thought.suffix, thought.output_tokens)
        } else {
            (framing, 0)
        };
        let stops = self.marker_tokens(&self.chat.answer_stops)?;
        let holdback = request.stop.iter().map(String::len).max().unwrap_or(1) - 1;
        let mut answer = Answer {
            trim_start: matches!(request.prompt, GenerationPrompt::Chat(_)),
            ..Answer::default()
        };
        let mut sink = |tokens: &[i32], tokenizer: &dyn TextTokenizer| {
            answer.tokens.extend(tokens);
            match tokenizer.decode(&answer.tokens) {
                Ok(text) => answer.advance(text, &request.stop, holdback, false, on_text),
                Err(error) => {
                    answer.error = Some(error);
                    false
                }
            }
        };
        let generated = self.generate_tokens(
            &prompt,
            &start,
            max_tokens,
            &stops,
            seed,
            self.decoding,
            &mut sink,
        )?;
        if let Some(error) = answer.error.take() {
            return Err(error);
        }
        if !answer.stopped && !answer.gone {
            let text = self.model.tokenizer().decode(&answer.tokens)?;
            answer.advance(text, &request.stop, holdback, true, on_text);
        }
        let finish = if generated.stopped || answer.stopped {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        Ok(Generation {
            text: answer.text,
            prompt_tokens: prompt_tokens + start.len(),
            completion_tokens: reasoning_tokens
                + generated.tokens.len()
                + usize::from(generated.stopped),
            reasoning_tokens,
            finish,
        })
    }

    /// The user turn with any images, then the opened model turn.
    fn prompt_parts(&mut self, text: &str, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
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

    fn read_canvas(
        &mut self,
        group: &[PreparedSlot],
        prompt_length: usize,
        seed: u64,
        steps: usize,
    ) -> Result<(Vec<SlotRead>, usize, f64)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut canvas = Vec::new();
        let mut slots = Vec::new();
        for slot in group {
            canvas.extend(&slot.prefix);
            let position = canvas.len();
            let initial_token = uniform::noise(self.n_vocab, self.mask, &mut rng);
            canvas.push(initial_token);
            slots.push(SlotRead {
                canvas_position: position,
                absolute_position: prompt_length + position,
                initial_token,
                candidate_tokens: slot.candidates.clone(),
                logits: Vec::new(),
                probabilities: Vec::new(),
            });
        }
        let positions: Vec<_> = slots.iter().map(|s| s.canvas_position).collect();
        let mut previous: Option<Vec<f32>> = None;
        let mut inverse_temperature = 1.0;
        let start = Instant::now();
        for step in 0..steps {
            let last = step + 1 == steps;
            self.model.forward_canvas(
                &canvas,
                prompt_length,
                conditioning(previous.as_deref(), inverse_temperature),
                if last {
                    Logits::Candidates
                } else {
                    Logits::Full
                },
            )?;
            if last {
                for slot in &mut slots {
                    slot.logits = self
                        .model
                        .candidate_logits(slot.canvas_position, &slot.candidate_tokens)?;
                    slot.probabilities = restricted_softmax(&slot.logits)?;
                }
            } else {
                let logits = self.model.full_logits()?;
                let temperature = 0.4 + 0.4 * (steps - step) as f64 / steps as f64;
                uniform::refine(
                    &mut canvas,
                    &logits,
                    &positions,
                    self.n_vocab as usize,
                    temperature,
                    self.mask,
                    &mut rng,
                )?;
                previous = Some(logits);
                inverse_temperature = (1.0 / temperature) as f32;
            }
        }
        Ok((slots, canvas.len(), start.elapsed().as_secs_f64() * 1000.0))
    }

    /// Masked-diffusion read: every answer position starts as the mask token. Each step commits
    /// the most confident position (by its probability over the candidates) plus any at or
    /// above `threshold`; the last step reads whatever is still masked. A slot reports the
    /// logits of the step that committed it, and committed rows are never read again.
    fn read_masked(
        &mut self,
        group: &[PreparedSlot],
        prompt_length: usize,
        steps: usize,
        threshold: f64,
    ) -> Result<(Vec<SlotRead>, usize, f64)> {
        let mut canvas = Vec::new();
        let mut slots = Vec::new();
        for slot in group {
            canvas.extend(&slot.prefix);
            let position = canvas.len();
            canvas.push(self.mask);
            slots.push(SlotRead {
                canvas_position: position,
                absolute_position: prompt_length + position,
                initial_token: self.mask,
                candidate_tokens: slot.candidates.clone(),
                logits: Vec::new(),
                probabilities: Vec::new(),
            });
        }
        // Trailing masks complete the block, the canvas shape the model was trained on.
        canvas.resize(self.padded(canvas.len()), self.mask);
        let mut pending: Vec<usize> = (0..slots.len()).collect();
        let start = Instant::now();
        for step in 0..steps {
            self.model.forward_canvas(
                &canvas,
                prompt_length,
                Conditioning::None,
                Logits::Candidates,
            )?;
            let mut reads = Vec::with_capacity(pending.len());
            for &i in &pending {
                let logits = self
                    .model
                    .candidate_logits(slots[i].canvas_position, &slots[i].candidate_tokens)?;
                let probabilities = restricted_softmax(&logits)?;
                reads.push((logits, probabilities));
            }
            let committed: Vec<usize> = if step + 1 == steps {
                (0..pending.len()).collect()
            } else {
                let proposals: Vec<_> = reads
                    .iter()
                    .map(|(_, p)| masked::Proposal {
                        token: best_index(p) as i32,
                        confidence: p[best_index(p)],
                    })
                    .collect();
                masked::commits(&proposals, threshold)
            };
            for &k in &committed {
                let slot = &mut slots[pending[k]];
                let (logits, probabilities) = std::mem::take(&mut reads[k]);
                canvas[slot.canvas_position] = slot.candidate_tokens[best_index(&probabilities)];
                slot.logits = logits;
                slot.probabilities = probabilities;
            }
            let mut index = 0;
            pending.retain(|_| {
                index += 1;
                !committed.contains(&(index - 1))
            });
            if pending.is_empty() {
                break;
            }
        }
        Ok((slots, canvas.len(), start.elapsed().as_secs_f64() * 1000.0))
    }

    /// The thought's opening tokens, closing tokens and single-token stop markers.
    fn thought_markers(&self) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let open = self.tokenize(&self.chat.thought_open, false, true)?;
        let close = self.tokenize(&self.chat.thought_close, false, true)?;
        let stops = self.marker_tokens(&self.chat.thought_stops)?;
        Ok((open, close, stops))
    }

    fn marker_tokens(&self, markers: &[String]) -> Result<Vec<i32>> {
        markers.iter().map(|m| self.single_token(m)).collect()
    }

    fn single_token(&self, marker: &str) -> Result<i32> {
        match self.tokenize(marker, false, true)?[..] {
            [token] => Ok(token),
            _ => Err(Error::UnsupportedModel(format!(
                "the tokenizer has no single token for the chat marker {marker}"
            ))),
        }
    }

    /// A bounded thought after `prompt`, generated with `decoding`.
    fn think_with(
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
    fn generate_tokens(
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

fn conditioning(previous: Option<&[f32]>, inverse_temperature: f32) -> Conditioning<'_> {
    match previous {
        Some(logits) => Conditioning::Previous {
            logits,
            inverse_temperature,
        },
        None => Conditioning::None,
    }
}

struct PreparedSlot {
    prefix: Vec<i32>,
    candidates: Vec<i32>,
}
/// Answer tokens generated when neither `max_tokens` nor the context gives a smaller limit.
const DEFAULT_MAX_TOKENS: usize = 2048;

/// The answer text of a generation as it grows, with what has been passed on.
#[derive(Default)]
struct Answer {
    tokens: Vec<i32>,
    /// Answer text, cut before any stop sequence.
    text: String,
    /// Bytes of `text` already passed on.
    emitted: usize,
    stopped: bool,
    gone: bool,
    error: Option<Error>,
    /// Chat answers drop the newlines models put after the closed thought.
    trim_start: bool,
}

impl Answer {
    /// Takes the decoded text so far, cuts it at the first stop sequence, and passes on what
    /// can no longer change: everything once `last` or stopped, otherwise all but `holdback`
    /// bytes and any incomplete character. Returns whether generation should continue.
    fn advance(
        &mut self,
        mut text: String,
        stop: &[String],
        holdback: usize,
        last: bool,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> bool {
        if self.trim_start {
            text = text.trim_start_matches('\n').to_string();
        }
        if let Some(cut) = stop.iter().filter_map(|s| text.find(s.as_str())).min() {
            text.truncate(cut);
            self.stopped = true;
        }
        let mut end = if last || self.stopped {
            text.len()
        } else {
            text.trim_end_matches('\u{FFFD}')
                .len()
                .saturating_sub(holdback)
        };
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        if end > self.emitted {
            if !on_text(&text[self.emitted..end]) {
                self.gone = true;
            }
            self.emitted = end;
        }
        self.text = text;
        !(self.stopped || self.gone)
    }
}

/// Receives each newly decided run of generated tokens; returns false to end generation.
type Sink<'a> = &'a mut dyn FnMut(&[i32], &dyn TextTokenizer) -> bool;

/// Tokens generated by one decoding run, before any stop marker.
#[derive(Default)]
struct Generated {
    tokens: Vec<i32>,
    /// A stop marker ended generation.
    stopped: bool,
    /// The sink ended generation.
    cancelled: bool,
    input_tokens: usize,
    forwards: usize,
    forward_ms: f64,
    /// Tokens decided by each causal verification.
    accepted: Vec<usize>,
}

impl Generated {
    /// Appends decided tokens and passes them on; false when generation should end.
    fn push(
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
struct Thought {
    suffix: Vec<i32>,
    input_tokens: usize,
    output_tokens: usize,
    forward_ms: f64,
    /// Generation forwards and tokens per causal verification, printed by the model tests.
    #[allow(dead_code)]
    forwards: usize,
    #[allow(dead_code)]
    accepted: Vec<usize>,
}

fn chunk_ranges(lengths: &[usize], capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let mut ranges = Vec::new();
    let (mut start, mut length) = (0, 0);
    for (i, &size) in lengths.iter().enumerate() {
        if size > capacity {
            return Err(Error::InvalidInput(format!(
                "A question's {size}-token template exceeds the {capacity}-token canvas"
            )));
        }
        if length + size > capacity {
            ranges.push(start..i);
            start = i;
            length = 0;
        }
        length += size;
    }
    if start < lengths.len() {
        ranges.push(start..lengths.len());
    }
    Ok(ranges)
}

fn best_index(probabilities: &[f64]) -> usize {
    probabilities.iter().enumerate().fold(
        0,
        |best, (i, p)| if *p > probabilities[best] { i } else { best },
    )
}

fn average_reads(mut reads: Vec<Vec<SlotRead>>) -> Result<Vec<SlotRead>> {
    let count = reads.len();
    if count == 0 {
        return Err(Error::InvalidLogits);
    }
    let mut means = reads.remove(0);
    for read in reads {
        if read.len() != means.len() {
            return Err(Error::InvalidLogits);
        }
        for (mean, slot) in means.iter_mut().zip(read) {
            if mean.candidate_tokens != slot.candidate_tokens
                || mean.probabilities.len() != slot.probabilities.len()
            {
                return Err(Error::InvalidLogits);
            }
            for (sum, p) in mean.probabilities.iter_mut().zip(slot.probabilities) {
                *sum += p;
            }
        }
    }
    for mean in &mut means {
        for p in &mut mean.probabilities {
            *p /= count as f64;
        }
        // An averaged distribution has no single native logit row.
        if count > 1 {
            mean.logits.clear();
        }
    }
    Ok(means)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;
    use crate::fake::{BOS, FakeModel, FakeTokenizer, Log, MASK};
    use jevons_core::{Message, TextTokenizer};

    #[cfg(feature = "models")]
    fn model_test_config() -> ModelConfig {
        ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap())
    }

    #[test]
    fn chunking_preserves_order_and_keeps_whole_questions() {
        assert_eq!(
            chunk_ranges(&[12, 20, 32, 1, 63], 64).unwrap(),
            vec![0..3, 3..5]
        );
        assert_eq!(chunk_ranges(&[64, 64], 64).unwrap(), vec![0..1, 1..2]);
        assert!(chunk_ranges(&[65], 64).is_err());
    }

    fn fake_engine(model: FakeModel) -> (Engine, std::rc::Rc<std::cell::RefCell<Log>>) {
        let log = model.log.clone();
        (Engine::new(Box::new(model)).unwrap(), log)
    }

    fn fake_request(slots: usize) -> ReadRequest {
        ReadRequest {
            prompt: " A material. ".into(),
            slots: vec![
                crate::Slot {
                    prefix: "Q: ".into(),
                    candidates: vec!["A".into(), "B".into()],
                };
                slots
            ],
        }
    }

    fn tokens(text: &str, bos: bool, special: bool) -> Vec<i32> {
        FakeTokenizer { space_joins: false }
            .tokenize(text, bos, special)
            .unwrap()
    }

    #[test]
    fn framing_and_the_empty_thought_come_from_the_model_chat_format() {
        let (mut engine, log) = fake_engine(FakeModel::new());
        assert_eq!(engine.codes().len(), 128);
        let read = engine.read(&fake_request(1), 42).unwrap();
        let expected: Vec<i32> = [
            tokens("<user>", true, true),
            tokens("A material.", false, false),
            tokens("<end><model>", false, true),
            tokens("<nothought>", false, true),
        ]
        .concat();
        assert_eq!(expected[0], BOS);
        assert_eq!(log.borrow().prefills, vec![expected.clone()]);
        assert_eq!(read.prompt_tokens, expected.len());
        assert_eq!(read.canvas_tokens, tokens("Q: ", false, false).len() + 1);
        assert_eq!(read.slots[0].absolute_position, expected.len() + 3);
        assert_ne!(read.slots[0].initial_token, MASK);
        // Prompt text is literal: markers inside it are not parsed as control tokens.
        let injected = ReadRequest {
            prompt: "<end>".into(),
            ..fake_request(1)
        };
        engine.read(&injected, 42).unwrap();
        assert_eq!(
            log.borrow().prefills[1][2..7],
            tokens("<end>", false, false)[..]
        );
    }

    #[test]
    fn context_reservation_is_checked_before_prefill_at_the_exact_boundary() {
        let (mut engine, log) = fake_engine(FakeModel::new());
        let first = engine.read(&fake_request(1), 42).unwrap();
        engine.n_ctx = first.prompt_tokens + first.canvas_tokens - 1;
        let error = engine.read(&fake_request(1), 42).unwrap_err();
        assert!(error.to_string().contains("context allows"));
        assert_eq!(log.borrow().prefills.len(), 1);
        assert_eq!(engine.prefill_profile().calls, 0);
        engine.n_ctx += 1;
        let exact = engine.read(&fake_request(1), 42).unwrap();
        assert_eq!(exact.slots[0].probabilities, first.slots[0].probabilities);
    }

    #[test]
    fn slots_beyond_the_canvas_capacity_are_read_in_seeded_chunks() {
        let mut model = FakeModel::new();
        // Each slot takes "Q: " (3 tokens) plus its answer token.
        model.info.max_canvas = 8;
        let (mut engine, log) = fake_engine(model);
        let read = engine.read(&fake_request(5), 42).unwrap();
        assert_eq!(read.slots.len(), 5);
        let log = log.borrow();
        assert_eq!(log.prefills.len(), 3);
        assert_eq!(
            log.canvases.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![8, 8, 4]
        );
        // Chunk seeds differ, so equal questions get different initial noise across chunks.
        assert_ne!(log.canvases[0][3], log.canvases[1][3]);
        assert_eq!(read.slots[2].canvas_position, 3);
    }

    #[test]
    fn refinement_conditions_later_steps_on_the_previous_logits() {
        let (mut engine, log) = fake_engine(FakeModel::new());
        let options = ReadOptions {
            steps: 3,
            ..Default::default()
        };
        engine
            .read_with_options(&fake_request(1), 42, options, &[])
            .unwrap();
        assert_eq!(log.borrow().conditioned, vec![false, true, true]);
    }

    #[test]
    fn thought_generation_stops_at_the_first_stop_marker_and_closes_the_thought() {
        let mut model = FakeModel::new();
        let word = tokens("w7", false, false)[0];
        model.favored = vec![word, tokens("<end>", false, true)[0], word];
        let (mut engine, log) = fake_engine(model);
        let options = ReadOptions {
            think: 8,
            ..Default::default()
        };
        let read = engine
            .read_with_options(&fake_request(1), 42, options, &[])
            .unwrap();
        assert_eq!(read.output_tokens, 2);
        let log = log.borrow();
        assert_eq!(log.canvases[0].len(), 8);
        let answer_prefill = log.prefills.last().unwrap();
        let thought = [
            tokens("<think>", false, true),
            vec![word],
            tokens("</think>", false, true),
        ]
        .concat();
        assert!(answer_prefill.ends_with(&thought));
    }

    fn masked_model() -> FakeModel {
        let mut model = FakeModel::new();
        model.scheme = DiffusionScheme::Masked {
            mask: MASK,
            block: 4,
            threshold: 0.9,
            max_steps: 4,
        };
        model
    }

    #[test]
    fn masked_reads_commit_the_most_confident_slot_each_step() {
        let (mut engine, log) = fake_engine(masked_model());
        let options = ReadOptions {
            steps: 2,
            ..Default::default()
        };
        let read = engine
            .read_with_options(&fake_request(3), 42, options, &[])
            .unwrap();
        let log = log.borrow();
        assert_eq!(log.canvases.len(), 2);
        let positions: Vec<_> = read.slots.iter().map(|s| s.canvas_position).collect();
        assert!(positions.iter().all(|&p| log.canvases[0][p] == MASK));
        assert!(read.slots.iter().all(|s| s.initial_token == MASK));
        // Candidate probabilities are equal across rows (0.73 < threshold), so only the first
        // slot commits at step one, as its best candidate; the last step reads the rest.
        assert_eq!(
            log.canvases[1][positions[0]],
            read.slots[0].candidate_tokens[0]
        );
        assert_eq!(log.canvases[1][positions[1]], MASK);
        assert_eq!(
            read.slots[0].logits,
            vec![positions[0] as f64 * 0.01, positions[0] as f64 * 0.01 - 1.0]
        );
        assert!(
            read.slots
                .iter()
                .all(|s| (s.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12)
        );
        assert!(log.conditioned.iter().all(|c| !c));
    }

    #[test]
    fn masked_thought_blocks_start_from_the_causal_prediction_and_stop_at_a_marker() {
        let mut model = masked_model();
        let word = tokens("w7", false, false)[0];
        model.causal = vec![word];
        model.favored = vec![0, word, tokens("</think>", false, true)[0], word];
        let (mut engine, log) = fake_engine(model);
        let options = ReadOptions {
            think: 16,
            ..Default::default()
        };
        let read = engine
            .read_with_options(&fake_request(1), 42, options, &[])
            .unwrap();
        assert_eq!(read.output_tokens, 3);
        let log = log.borrow();
        // The block starts with the causal prediction; one confident step fills its masks.
        // The second canvas is the answer read.
        assert_eq!(log.canvases.len(), 2);
        assert_eq!(log.canvases[0], vec![word, MASK, MASK, MASK]);
        assert_eq!(log.predicted, vec![1]);
        let thought = [
            tokens("<think>", false, true),
            vec![word, word],
            tokens("</think>", false, true),
        ]
        .concat();
        assert!(log.prefills.last().unwrap().ends_with(&thought));
    }

    /// Thought tokens of a speculative or autoregressive fake run, with the engine log.
    fn causal_thought(
        decoding: Decoding,
        favored: Vec<i32>,
        think: usize,
    ) -> (ReadResult, std::rc::Rc<std::cell::RefCell<Log>>, Vec<i32>) {
        let mut model = masked_model();
        let script: Vec<i32> = (1..=5)
            .map(|i| tokens(&format!("w{i}"), false, false)[0])
            .collect();
        model.causal = [script.clone(), tokens("</think>", false, true)].concat();
        model.favored = favored;
        let (mut engine, log) = fake_engine(model);
        engine.set_decoding(decoding).unwrap();
        let options = ReadOptions {
            think,
            ..Default::default()
        };
        let read = engine
            .read_with_options(&fake_request(1), 42, options, &[])
            .unwrap();
        (read, log, script)
    }

    #[test]
    fn self_speculation_keeps_matching_drafts_and_the_next_causal_token() {
        let w = |i: usize| tokens(&format!("w{i}"), false, false)[0];
        let off_script = w(40);
        // Drafts at block rows 1..3: the next two script tokens, then a miss.
        let (read, log, script) = causal_thought(
            Decoding::SelfSpeculation,
            vec![0, w(2), w(3), off_script],
            16,
        );
        assert_eq!(read.output_tokens, script.len() + 1);
        let log = log.borrow();
        // The first draft keeps w2 and w3, then takes the causal w4 over the miss; later drafts
        // miss at once and keep only the causal token, the last of which closes the thought.
        // The fourth canvas is the answer read.
        assert_eq!(log.canvases.len(), 4);
        assert_eq!(
            log.canvases[..3],
            [
                vec![w(1), MASK, MASK, MASK],
                vec![w(4), MASK, MASK, MASK],
                vec![w(5), MASK, MASK, MASK],
            ]
        );
        assert_eq!(log.predicted, vec![1, 4, 4, 4]);
        let open = tokens("<think>", false, true);
        let first_verify = [open.clone(), vec![w(1), w(2), w(3), off_script]].concat();
        assert!(log.prefills.iter().any(|p| p.ends_with(&first_verify)));
        let thought = [open, script, tokens("</think>", false, true)].concat();
        assert!(log.prefills.last().unwrap().ends_with(&thought));
    }

    #[test]
    fn autoregressive_and_speculative_thoughts_agree_and_respect_the_budget() {
        let w = |i: usize| tokens(&format!("w{i}"), false, false)[0];
        for think in [3, 16] {
            let (ar, ar_log, script) = causal_thought(Decoding::Autoregressive, Vec::new(), think);
            let (spec, spec_log, _) =
                causal_thought(Decoding::SelfSpeculation, vec![0, w(2), w(9), w(9)], think);
            assert_eq!(ar.output_tokens, spec.output_tokens);
            assert_eq!(
                ar_log.borrow().prefills.last(),
                spec_log.borrow().prefills.last()
            );
            assert!(ar_log.borrow().canvases.len() <= 1, "only the answer read");
            let expected = if think == 3 { 3 } else { script.len() + 1 };
            assert_eq!(ar.output_tokens, expected);
        }
    }

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

    /// A causal fake engine whose answer is `w1 w2 ...` (as scripted) and a generation request.
    fn generation(
        causal: Vec<i32>,
        prompt: GenerationPrompt,
        max_tokens: Option<usize>,
        stop: &[&str],
    ) -> (
        Result<Generation>,
        Vec<String>,
        std::rc::Rc<std::cell::RefCell<Log>>,
    ) {
        let mut model = masked_model();
        model.causal = causal;
        let (mut engine, log) = fake_engine(model);
        engine.set_decoding(Decoding::Autoregressive).unwrap();
        let request = GenerationRequest {
            prompt,
            max_tokens,
            think: 0,
            stop: stop.iter().map(|s| s.to_string()).collect(),
        };
        let mut deltas = Vec::new();
        let result = engine.generate(&request, 42, &mut |text| {
            deltas.push(text.to_string());
            true
        });
        (result, deltas, log)
    }

    fn words(n: usize) -> Vec<i32> {
        (1..=n)
            .map(|i| tokens(&format!("w{i}"), false, false)[0])
            .collect()
    }

    fn chat(text: &str) -> GenerationPrompt {
        GenerationPrompt::Chat(vec![
            Message {
                role: Role::System,
                text: "Be brief.".into(),
            },
            Message {
                role: Role::User,
                text: text.into(),
            },
        ])
    }

    #[test]
    fn chat_generation_frames_every_turn_and_ends_at_the_turn_marker() {
        let end = tokens("<end>", false, true)[0];
        let newline = tokens("\n", false, false)[0];
        let (result, deltas, log) = generation(
            [vec![newline], words(3), vec![end]].concat(),
            chat("Hi"),
            None,
            &[],
        );
        let answer = result.unwrap();
        assert_eq!(answer.text, "w1w2w3");
        assert_eq!(deltas.concat(), "w1w2w3");
        assert_eq!(answer.finish, FinishReason::Stop);
        assert_eq!(answer.completion_tokens, 5);
        let framed = [
            tokens("<system>", true, true),
            tokens("Be brief.", false, false),
            tokens("<end><user>", false, true),
            tokens("Hi", false, false),
            tokens("<end><model><nothought>", false, true),
        ]
        .concat();
        assert_eq!(answer.prompt_tokens, framed.len());
        assert_eq!(log.borrow().prefills[0], framed);
    }

    #[test]
    fn stop_sequences_cut_the_answer_and_are_never_streamed_in_part() {
        let (result, deltas, _) = generation(words(6), chat("Hi"), Some(16), &["w3w", "zz"]);
        let answer = result.unwrap();
        assert_eq!(answer.text, "w1w2");
        assert_eq!(answer.finish, FinishReason::Stop);
        assert_eq!(deltas.concat(), "w1w2");
        assert!(
            deltas.len() > 1,
            "text is streamed as it is decided: {deltas:?}"
        );
    }

    #[test]
    fn answers_end_at_max_tokens_and_text_prompts_are_not_framed() {
        let (result, _, log) = generation(
            words(6),
            GenerationPrompt::Text("Once".into()),
            Some(2),
            &[],
        );
        let answer = result.unwrap();
        assert_eq!(answer.text, "w1w2");
        assert_eq!(answer.finish, FinishReason::Length);
        assert_eq!(log.borrow().prefills[0], tokens("Once", true, false));
        // The prompt and answer must fit the 256-token context.
        let (result, _, _) = generation(words(1), chat("Hi"), Some(1000), &[]);
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn a_client_that_stops_listening_ends_generation() {
        let mut model = masked_model();
        model.causal = words(20);
        let (mut engine, log) = fake_engine(model);
        engine.set_decoding(Decoding::Autoregressive).unwrap();
        let request = GenerationRequest {
            prompt: chat("Hi"),
            max_tokens: Some(16),
            think: 0,
            stop: Vec::new(),
        };
        let mut calls = 0;
        engine
            .generate(&request, 42, &mut |_| {
                calls += 1;
                false
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(log.borrow().predicted.len(), 1);
    }

    #[test]
    fn masked_canvases_pad_to_the_block_and_answers_carry_the_prefix_space() {
        let mut model = masked_model();
        model.scheme = DiffusionScheme::Masked {
            mask: MASK,
            block: 8,
            threshold: 0.9,
            max_steps: 8,
        };
        model.chat.space_joins_answers = true;
        model.tokenizer.space_joins = true;
        let spaced = |text: &str| {
            FakeTokenizer { space_joins: true }
                .tokenize(text, false, false)
                .unwrap()
        };
        let (mut engine, log) = fake_engine(model);
        assert_eq!(engine.codes().len(), 128);
        let read = engine.read(&fake_request(1), 42).unwrap();
        // "Q: " loses its trailing space; candidates become " A" and " B".
        assert_eq!(
            read.slots[0].candidate_tokens,
            [spaced(" A")[0], spaced(" B")[0]]
        );
        assert_eq!(read.slots[0].canvas_position, 2);
        let log = log.borrow();
        let mut expected = spaced("Q:");
        expected.extend([MASK; 6]);
        assert_eq!(log.canvases[0], expected);
        assert_eq!(read.canvas_tokens, 8);
    }

    #[test]
    fn samples_average_probabilities_instead_of_logits_or_winning_labels() {
        let slot = |probabilities: Vec<f64>| SlotRead {
            canvas_position: 0,
            absolute_position: 1,
            initial_token: 7,
            candidate_tokens: vec![1, 2],
            logits: vec![9.0, 1.0],
            probabilities,
        };
        let means = average_reads(vec![
            vec![slot(vec![0.99, 0.01])],
            vec![slot(vec![0.25, 0.75])],
        ])
        .unwrap();
        assert_eq!(means[0].probabilities, vec![0.62, 0.38]);
        assert!(means[0].logits.is_empty());
    }

    #[cfg(feature = "models")]
    #[test]
    #[ignore = "Requires DIFFUSION_MODEL and a HIP GPU"]
    fn model_extensions_average_refine_think_and_chunk() {
        let config = model_test_config();
        let mut engine = Engine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag is used in concrete.");
        let first = engine.read(&request, 42).unwrap();
        let profile = engine.prefill_profile();
        assert_eq!(profile.calls, 1);
        assert_eq!(profile.processed_tokens, first.prompt_tokens);
        assert_eq!(profile.reused_tokens, 0);
        assert!(profile.batches > 0);
        assert!(profile.wall_ms.is_finite() && profile.wall_ms > 0.0);
        let second = engine.read(&request, 42 + 7919).unwrap();
        let averaged = engine
            .read_with_options(
                &request,
                42,
                ReadOptions {
                    samples: 2,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        for (i, &p) in averaged.slots[0].probabilities.iter().enumerate() {
            assert!(
                (p - (first.slots[0].probabilities[i] + second.slots[0].probabilities[i]) / 2.0)
                    .abs()
                    < 1e-5
            );
        }
        assert_eq!(averaged.prompt_tokens, first.prompt_tokens * 2);
        // Samples reuse the same prefill within this read; logical usage counts both.
        // A backend with a prompt cache may serve the repeated prompt without recomputing it.
        let profile = engine.prefill_profile();
        assert_eq!(profile.calls, 1);
        assert_eq!(
            profile.processed_tokens + profile.reused_tokens,
            first.prompt_tokens
        );
        assert_eq!(averaged.canvas_tokens, first.canvas_tokens * 2);
        let refined = engine
            .read_with_options(
                &request,
                42,
                ReadOptions {
                    steps: 3,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        assert!((refined.slots[0].probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        let thought = engine
            .read_with_options(
                &request,
                42,
                ReadOptions {
                    think: 8,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        assert!((1..=8).contains(&thought.output_tokens));
        assert!(thought.prompt_tokens > first.prompt_tokens);
        let many = ReadRequest {
            prompt: request.prompt.clone(),
            slots: vec![request.slots[0].clone(); 12],
        };
        for sequential in [false, true] {
            let read = engine
                .read_with_options(
                    &many,
                    42,
                    ReadOptions {
                        sequential,
                        ..Default::default()
                    },
                    &[],
                )
                .unwrap();
            assert_eq!(read.slots.len(), 12);
            assert_eq!(read.canvas_tokens, first.canvas_tokens * 12);
            assert!(
                read.slots
                    .iter()
                    .all(|slot| (slot.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6)
            );
        }
        let repeated = engine.read(&request, 42).unwrap();
        for (a, b) in first.slots[0]
            .probabilities
            .iter()
            .zip(&repeated.slots[0].probabilities)
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Extension state leaked into the next request"
            );
        }
        let error = engine
            .read_with_options(
                &request,
                42,
                ReadOptions::default(),
                &[ImageInput { bytes: vec![1] }],
            )
            .unwrap_err();
        assert!(error.to_string().contains("--mmproj"));
    }

    #[cfg(feature = "models")]
    #[test]
    #[ignore = "Requires DIFFUSION_MODEL, DIFFUSION_MMPROJ and a HIP GPU"]
    fn model_images_prefill_and_preserve_text_reproducibility() {
        let mut config = ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap());
        config.mmproj = Some(std::env::var("DIFFUSION_MMPROJ").unwrap().into());
        let mut engine = Engine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag.");
        let before = engine.read(&request, 42).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(224, 224, image::Rgb([255, 0, 0]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let question = ReadRequest {
            prompt: "What color is the image? A = red, B = blue".into(),
            slots: vec![crate::Slot {
                prefix: "Answer: ".into(),
                candidates: vec!["A".into(), "B".into()],
            }],
        };
        let image = ImageInput {
            bytes: png.into_inner(),
        };
        let read = engine
            .read_with_options(
                &question,
                42,
                ReadOptions {
                    steps: 2,
                    samples: 2,
                    ..Default::default()
                },
                std::slice::from_ref(&image),
            )
            .unwrap();
        assert!(read.prompt_tokens > 100);
        assert!(read.slots[0].probabilities[0] > read.slots[0].probabilities[1]);
        // The same image again reuses its cached prompt block and reproduces the read exactly.
        let again = engine
            .read_with_options(
                &question,
                42,
                ReadOptions {
                    steps: 2,
                    samples: 2,
                    ..Default::default()
                },
                std::slice::from_ref(&image),
            )
            .unwrap();
        assert!(engine.prefill_profile().reused_tokens > 100);
        assert_eq!(again.slots[0].probabilities, read.slots[0].probabilities);
        let after = engine.read(&request, 42).unwrap();
        for (a, b) in before.slots[0]
            .probabilities
            .iter()
            .zip(&after.slots[0].probabilities)
        {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[cfg(feature = "models")]
    #[test]
    #[ignore = "Requires DIFFUSION_MODEL and a HIP GPU"]
    fn model_reads_preserve_reproducibility_across_requests() {
        let config = model_test_config();
        let mut engine = Engine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag is used in concrete.");
        let first = engine.read(&request, 42).unwrap();
        let expected_prompt_tokens: usize = [
            ("<|turn>user\n", true, true),
            (request.prompt.trim(), false, false),
            ("<turn|>\n<|turn>model\n", false, true),
            ("<|channel>thought\n<channel|>", false, true),
        ]
        .into_iter()
        .map(|(text, special, parse)| engine.tokenize(text, special, parse).unwrap().len())
        .sum();
        assert_eq!(first.prompt_tokens, expected_prompt_tokens);
        assert_eq!(first.output_tokens, 0);
        assert_eq!(
            first.slots[0].absolute_position,
            expected_prompt_tokens + first.slots[0].canvas_position
        );

        // Lower the logical limit inside the allocated native context to check
        // that framing is reserved before inference, including the exact boundary.
        let context_size = engine.n_ctx;
        engine.n_ctx = first.prompt_tokens + first.canvas_tokens - 1;
        let error = engine.read(&request, 42).unwrap_err();
        assert!(error.to_string().contains("context allows"));
        assert_eq!(engine.prefill_profile().calls, 0);
        assert_eq!(engine.prefill_profile().processed_tokens, 0);
        engine.n_ctx += 1;
        let exact_fit = engine.read(&request, 42).unwrap();
        assert_eq!(exact_fit.prompt_tokens, first.prompt_tokens);
        assert_eq!(
            exact_fit.slots[0].probabilities,
            first.slots[0].probabilities
        );
        engine.n_ctx = context_size;
        let other = ReadRequest::scm(
            "This is a different and longer material description: steel reinforcement bars carry tensile forces in a reinforced concrete structure.",
        );
        engine.read(&other, 7).unwrap();
        let repeated = engine.read(&request, 42).unwrap();
        let profile = engine.prefill_profile();
        assert_eq!(profile.calls, 1);
        assert_eq!(
            profile.processed_tokens + profile.reused_tokens,
            first.prompt_tokens
        );
        assert_eq!(
            first.slots[0].initial_token,
            repeated.slots[0].initial_token
        );
        assert_eq!(first.slots[0].candidate_tokens.len(), 2);
        assert_eq!(first.canvas_tokens, 12);
        assert_eq!(first.slots[0].canvas_position, 11);
        for (a, b) in first.slots[0]
            .probabilities
            .iter()
            .zip(&repeated.slots[0].probabilities)
        {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[cfg(feature = "models")]
    fn nemotron_engine() -> Engine {
        let mut config = ModelConfig::new(std::env::var("NEMOTRON_MODEL").unwrap());
        config.context_size = 4096;
        Engine::load(&config).unwrap()
    }

    #[cfg(feature = "models")]
    #[test]
    #[ignore = "Requires NEMOTRON_MODEL and a HIP GPU"]
    fn nemotron_reads_are_calibrated_reproducible_and_support_extensions() {
        let mut engine = nemotron_engine();
        assert_eq!(engine.model_info().architecture, "nemotron-diffusion");
        let scm = ReadRequest::scm("Ground granulated blast furnace slag is used in concrete.");
        let first = engine.read(&scm, 42).unwrap();
        // Answer tokens carry the prefix space (" A") and the canvas is padded to the block.
        assert_eq!(first.canvas_tokens, 32);
        assert_eq!(first.slots[0].initial_token, engine.mask);
        let yes = first.slots[0].probabilities[0];
        let rebar = engine
            .read(&ReadRequest::scm("Steel reinforcement bars."), 42)
            .unwrap();
        let rebar = rebar.slots[0].probabilities[0];
        println!("P(yes): slag {yes:.3}, rebar {rebar:.3}");
        // Every size reads slag as an SCM; the 3B is less sure that rebar is not one.
        assert!(yes > 0.7, "slag is an SCM: P(yes) = {yes}");
        assert!(rebar < yes - 0.2, "rebar is not an SCM: P(yes) = {rebar}");
        // The shared prompt prefix is reused, and the read is reproduced up to the small
        // differences of attention that is not row invariant after partial reuse (below 1e-3).
        let again = engine.read(&scm, 7).unwrap();
        let profile = engine.prefill_profile();
        assert!(profile.reused_tokens > 0);
        assert_eq!(
            profile.processed_tokens + profile.reused_tokens,
            first.prompt_tokens
        );
        for (a, b) in first.slots[0]
            .probabilities
            .iter()
            .zip(&again.slots[0].probabilities)
        {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
        let many = ReadRequest {
            prompt: scm.prompt.clone(),
            slots: vec![scm.slots[0].clone(); 6],
        };
        for (steps, sequential) in [(1, false), (3, false), (1, true)] {
            let options = ReadOptions {
                steps,
                sequential,
                ..Default::default()
            };
            let read = engine.read_with_options(&many, 42, options, &[]).unwrap();
            assert_eq!(read.slots.len(), 6);
            assert!(
                read.slots
                    .iter()
                    .all(|s| (s.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6)
            );
        }
        let options = ReadOptions {
            think: 40,
            ..Default::default()
        };
        let thought = engine.read_with_options(&scm, 42, options, &[]).unwrap();
        println!(
            "think: {} tokens, P(yes) {:.3}",
            thought.output_tokens, thought.slots[0].probabilities[0]
        );
        assert!((1..=40).contains(&thought.output_tokens));
        assert!(thought.prompt_tokens > first.prompt_tokens);
    }

    /// Thought generation on the GPU: self-speculation must give the autoregressive tokens.
    /// Prints tokens per forward and speed for every decoding (`--nocapture`); set
    /// `THINK_BUDGET` to change the 128-token budget.
    #[cfg(feature = "models")]
    #[test]
    #[ignore = "Requires NEMOTRON_MODEL and a HIP GPU"]
    fn nemotron_self_speculation_reproduces_autoregressive_thoughts() {
        let mut engine = nemotron_engine();
        assert!(matches!(engine.scheme, DiffusionScheme::Masked { .. }));
        let budget: usize = std::env::var("THINK_BUDGET")
            .map(|b| b.parse().unwrap())
            .unwrap_or(128);
        let prompts = [
            "What is 15% of 240? Explain the calculation.",
            "Is ground granulated blast furnace slag a supplementary cementitious material? Explain briefly.",
            "Which team should handle this request: \"Compute the least common multiple of 12 and 18\"? The teams are math, coding_agent and writing.",
        ];
        let run = |engine: &mut Engine, prompt: &[PromptPart], decoding| {
            let start = Instant::now();
            let thought = engine.think_with(prompt, budget, 42, decoding).unwrap();
            (thought, start.elapsed().as_secs_f64())
        };
        // Compile and tune kernels for every shape before timing.
        let warm = engine.prompt_parts(prompts[0], &[]).unwrap();
        for decoding in Decoding::ALL {
            run(&mut engine, &warm, decoding);
        }
        let mut totals = [(0usize, 0usize, 0.0f64); 3];
        for text in prompts {
            let prompt = engine.prompt_parts(text, &[]).unwrap();
            let mut suffixes = Vec::new();
            for (i, decoding) in Decoding::ALL.into_iter().enumerate() {
                let (thought, seconds) = run(&mut engine, &prompt, decoding);
                let accepted = &thought.accepted;
                println!(
                    "{:>16}: {:3} tokens, {:3} forwards ({:.2} tokens/forward, mean verified {:.2}), {:6.0} ms, {:5.1} tokens/s",
                    decoding.id(),
                    thought.output_tokens,
                    thought.forwards,
                    thought.output_tokens as f64 / thought.forwards as f64,
                    accepted.iter().sum::<usize>() as f64 / accepted.len().max(1) as f64,
                    seconds * 1000.0,
                    thought.output_tokens as f64 / seconds,
                );
                totals[i].0 += thought.output_tokens;
                totals[i].1 += thought.forwards;
                totals[i].2 += seconds;
                if decoding == Decoding::SelfSpeculation {
                    println!("  tokens {:?}", thought.suffix);
                }
                suffixes.push(thought.suffix);
            }
            let agreed = suffixes[1]
                .iter()
                .zip(&suffixes[2])
                .take_while(|(a, b)| a == b)
                .count();
            println!("  self-speculation matches autoregressive for {agreed} tokens");
            assert_eq!(suffixes[1], suffixes[2], "{text}");
        }
        for (decoding, (tokens, forwards, seconds)) in Decoding::ALL.into_iter().zip(totals) {
            println!(
                "total {:>16}: {tokens} tokens, {:.2} tokens/forward, {:.1} tokens/s",
                decoding.id(),
                tokens as f64 / forwards as f64,
                tokens as f64 / seconds
            );
        }
    }

    #[cfg(feature = "models")]
    #[test]
    #[ignore = "Requires NEMOTRON_MODEL and a HIP GPU"]
    fn nemotron_images_are_read_and_their_prompt_blocks_reused() {
        let mut engine = nemotron_engine();
        let png = |rgb: [u8; 3]| {
            let mut out = std::io::Cursor::new(Vec::new());
            image::RgbImage::from_pixel(224, 224, image::Rgb(rgb))
                .write_to(&mut out, image::ImageFormat::Png)
                .unwrap();
            ImageInput {
                bytes: out.into_inner(),
            }
        };
        let question = ReadRequest {
            prompt: "What color is the image? Use these answer codes:\nA = red\nB = blue".into(),
            slots: vec![crate::Slot {
                prefix: "Answer: ".into(),
                candidates: vec!["A".into(), "B".into()],
            }],
        };
        let red = engine
            .read_with_options(&question, 42, ReadOptions::default(), &[png([230, 20, 20])])
            .unwrap();
        let blue = engine
            .read_with_options(&question, 42, ReadOptions::default(), &[png([20, 40, 230])])
            .unwrap();
        println!(
            "P(red): red image {:.3}, blue image {:.3}",
            red.slots[0].probabilities[0], blue.slots[0].probabilities[0]
        );
        assert!(red.slots[0].probabilities[0] > 0.5);
        assert!(blue.slots[0].probabilities[0] < 0.5);
        // 224x224 is 8x8 merged tokens: start, 8 rows of 8 pads and a break (the last an end).
        assert!(red.prompt_tokens > 1 + 8 * 9);
        let again = engine
            .read_with_options(&question, 42, ReadOptions::default(), &[png([20, 40, 230])])
            .unwrap();
        assert!(engine.prefill_profile().reused_tokens > 8 * 9);
        for (a, b) in blue.slots[0]
            .probabilities
            .iter()
            .zip(&again.slots[0].probabilities)
        {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }
}
