//! Token preparation, repeated reads, and bounded diffusion generation.

use crate::sampler::{masked, uniform};
use crate::{Error, ReadRequest, ReadResult, Result, SlotRead, restricted_softmax};
use crate::{ImageInput, ReadOptions};
use jevons_core::{ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Logits, PromptPart};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::{collections::HashSet, time::Instant};

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
            chat,
            model,
            codes: Vec::new(),
        };
        engine.codes = engine.find_codes(128)?;
        Ok(engine)
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
        let mut prompt = vec![PromptPart::Text(self.tokenize(
            &self.chat.user_open,
            self.chat.bos,
            true,
        )?)];
        prompt.extend(self.model.encode_images(images)?);
        prompt.push(PromptPart::Text(self.tokenize(
            request.prompt.trim(),
            false,
            false,
        )?));
        prompt.push(PromptPart::Text(self.tokenize(
            &self.chat.model_open,
            false,
            true,
        )?));
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
            let thought = match self.scheme {
                DiffusionScheme::UniformSelfConditioned { .. } => {
                    self.think(&prompt, options.think, seed)?
                }
                DiffusionScheme::Masked {
                    block,
                    threshold,
                    max_steps,
                    ..
                } => self.think_masked(&prompt, options.think, block, threshold, max_steps)?,
            };
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

    /// Masked block generation: each block of mask tokens is unmasked by full-vocabulary
    /// confidence, truncated at the first stop marker, and committed to the prompt cache by the
    /// next prefill.
    fn think_masked(
        &mut self,
        prompt: &[PromptPart],
        budget: usize,
        block: usize,
        threshold: f64,
        max_steps: usize,
    ) -> Result<Thought> {
        let mut suffix = self.tokenize(&self.chat.thought_open, false, true)?;
        let close = self.tokenize(&self.chat.thought_close, false, true)?;
        let stops = self
            .chat
            .thought_stops
            .iter()
            .map(|marker| self.single_token(marker))
            .collect::<Result<Vec<_>>>()?;
        let vocab = self.n_vocab as usize;
        let (mut output_tokens, mut input_tokens, mut forward_ms) = (0, 0, 0.0);
        while output_tokens < budget {
            let prompt_length = self.model.prefill(prompt, &suffix)?;
            input_tokens += prompt_length;
            let count = block
                .min(self.max_canvas)
                .min(self.batch_size)
                .min(budget - output_tokens);
            let mut canvas = vec![self.mask; count];
            let mut open = vec![true; count];
            let start = Instant::now();
            for step in 0..max_steps.max(1) {
                self.model.forward_canvas(
                    &canvas,
                    prompt_length,
                    Conditioning::None,
                    Logits::Full,
                )?;
                let logits = self.model.full_logits()?;
                if logits.len() != count * vocab {
                    return Err(Error::InvalidLogits);
                }
                let rows: Vec<usize> = (0..count).filter(|&i| open[i]).collect();
                let proposals = rows
                    .iter()
                    .map(|&i| masked::propose(&logits[i * vocab..(i + 1) * vocab]))
                    .collect::<Result<Vec<_>>>()?;
                let committed = if step + 1 == max_steps.max(1) {
                    (0..rows.len()).collect()
                } else {
                    masked::commits(&proposals, threshold)
                };
                for k in committed {
                    canvas[rows[k]] = proposals[k].token;
                    open[rows[k]] = false;
                }
                // Stop once a stop marker is fixed and everything before it is too.
                let settled = open.iter().position(|&o| o).unwrap_or(count);
                if settled == count || canvas[..settled].iter().any(|t| stops.contains(t)) {
                    break;
                }
            }
            forward_ms += start.elapsed().as_secs_f64() * 1000.0;
            let settled = open.iter().position(|&o| o).unwrap_or(count);
            let stop = canvas[..settled].iter().position(|t| stops.contains(t));
            let length = stop.unwrap_or(settled);
            suffix.extend(&canvas[..length]);
            output_tokens += length + usize::from(stop.is_some());
            if stop.is_some() || length == 0 {
                break;
            }
        }
        suffix.extend(close);
        Ok(Thought {
            suffix,
            input_tokens,
            output_tokens,
            forward_ms,
        })
    }

    fn single_token(&self, marker: &str) -> Result<i32> {
        match self.tokenize(marker, false, true)?[..] {
            [token] => Ok(token),
            _ => Err(Error::UnsupportedModel(format!(
                "the tokenizer has no single token for the chat marker {marker}"
            ))),
        }
    }

    fn think(&mut self, prompt: &[PromptPart], budget: usize, seed: u64) -> Result<Thought> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut suffix = self.tokenize(&self.chat.thought_open, false, true)?;
        let close = self.tokenize(&self.chat.thought_close, false, true)?;
        let stops = self
            .chat
            .thought_stops
            .iter()
            .map(|marker| self.single_token(marker))
            .collect::<Result<Vec<_>>>()?;
        let mut output_tokens = 0;
        let mut input_tokens = 0;
        let mut forward_ms = 0.0;
        while output_tokens < budget {
            let prompt_length = self.model.prefill(prompt, &suffix)?;
            input_tokens += prompt_length;
            let count = self
                .max_canvas
                .min(self.batch_size)
                .min(budget - output_tokens);
            let mut canvas: Vec<_> = (0..count)
                .map(|_| uniform::noise(self.n_vocab, self.mask, &mut rng))
                .collect();
            let positions: Vec<_> = (0..count).collect();
            let mut previous: Option<Vec<f32>> = None;
            let mut previous_best = Vec::new();
            let mut inverse_temperature = 1.0;
            let start = Instant::now();
            for step in 0..48 {
                self.model.forward_canvas(
                    &canvas,
                    prompt_length,
                    conditioning(previous.as_deref(), inverse_temperature),
                    Logits::Full,
                )?;
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
            forward_ms += start.elapsed().as_secs_f64() * 1000.0;
            let stop = previous_best.iter().position(|t| stops.contains(t));
            let length = stop.unwrap_or(previous_best.len());
            suffix.extend(&previous_best[..length]);
            output_tokens += length + usize::from(stop.is_some());
            if stop.is_some() {
                break;
            }
        }
        suffix.extend(close);
        Ok(Thought {
            suffix,
            input_tokens,
            output_tokens,
            forward_ms,
        })
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
struct Thought {
    suffix: Vec<i32>,
    input_tokens: usize,
    output_tokens: usize,
    forward_ms: f64,
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
    use jevons_core::TextTokenizer;

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
    fn masked_thought_blocks_stop_at_a_marker_and_commit_through_prefill() {
        let mut model = masked_model();
        let word = tokens("w7", false, false)[0];
        model.favored = vec![word, word, tokens("</think>", false, true)[0], word];
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
        // One confident step fills the 4-token block of masks.
        assert_eq!(log.canvases[0], vec![MASK; 4]);
        let thought = [
            tokens("<think>", false, true),
            vec![word, word],
            tokens("</think>", false, true),
        ]
        .concat();
        assert!(log.prefills.last().unwrap().ends_with(&thought));
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
        assert!(yes > 0.7, "slag is an SCM: P(yes) = {yes}");
        let rebar = engine
            .read(&ReadRequest::scm("Steel reinforcement bars."), 42)
            .unwrap();
        assert!(rebar.slots[0].probabilities[0] < 0.5);
        println!(
            "P(yes): slag {yes:.3}, rebar {:.3}",
            rebar.slots[0].probabilities[0]
        );
        // The shared prompt prefix is reused, and the read is reproduced.
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
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
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
}
