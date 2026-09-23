//! Token preparation, repeated reads, and bounded diffusion generation.

use crate::backend::{Backend, PromptPart};
use crate::{Error, ModelConfig, ReadRequest, ReadResult, Result, SlotRead, restricted_softmax};
use crate::{ImageInput, ReadOptions, denoise};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::{collections::HashSet, time::Instant};

const EMPTY_THOUGHT_CHANNEL: &str = "<|channel>thought\n<channel|>";

pub struct Engine {
    backend: Box<dyn Backend>,
    codes: Vec<String>,
    n_vocab: i32,
    mask: i32,
    n_ctx: usize,
    batch_size: usize,
}

impl Engine {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        let backend: Box<dyn Backend> = Box::new(crate::cubecl::CubeBackend::load(config)?);
        for marker in ["<|turn>", "<turn|>"] {
            let tokens = backend.tokenize(marker, false, true)?;
            if tokens.len() != 1 {
                return Err(Error::InvalidInput(format!(
                    "The model does not have the DiffusionGemma text chat marker {marker}"
                )));
            }
        }
        let mut engine = Self {
            n_vocab: backend.n_vocab(),
            mask: backend.mask(),
            n_ctx: backend.n_ctx(),
            batch_size: backend.batch_size(),
            backend,
            codes: Vec::new(),
        };
        engine.codes = engine.find_codes(128)?;
        Ok(engine)
    }

    /// Verified single-token codes used to represent arbitrary external labels.
    pub fn codes(&self) -> &[String] {
        &self.codes
    }

    /// Synchronized prefill work for the last read; independent of logical usage.
    pub fn prefill_profile(&mut self) -> crate::PrefillProfile {
        *self.backend.profile()
    }

    fn find_codes(&self, count: usize) -> Result<Vec<String>> {
        let mut codes = Vec::with_capacity(count);
        let mut seen = HashSet::new();
        for ch in ('A'..='Z').chain('a'..='z').chain('0'..='9') {
            let code = ch.to_string();
            let tokens = self.backend.tokenize(&code, false, false)?;
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
            if let Some(code) = self.backend.code_piece(token)
                && self.backend.tokenize(&code, false, false)? == [token]
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
        *self.backend.profile() = crate::PrefillProfile::default();
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
        let mut prompt = vec![PromptPart::Text(self.backend.tokenize(
            "<|turn>user\n",
            true,
            true,
        )?)];
        prompt.extend(self.backend.image_parts(images)?);
        prompt.push(PromptPart::Text(self.backend.tokenize(
            request.prompt.trim(),
            false,
            false,
        )?));
        prompt.push(PromptPart::Text(self.backend.tokenize(
            "<turn|>\n<|turn>model\n",
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
                    let tokens = self.backend.tokenize(candidate, false, false)?;
                    if tokens.len() != 1 || candidates.contains(&tokens[0]) {
                        return Err(Error::InvalidInput(format!(
                            "Candidate {candidate:?} must encode to one distinct token"
                        )));
                    }
                    candidates.push(tokens[0]);
                }
                Ok(PreparedSlot {
                    prefix: self.backend.tokenize(&slot.prefix, false, false)?,
                    candidates,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let capacity = 64.min(self.batch_size);
        let lengths: Vec<_> = prepared.iter().map(|s| s.prefix.len() + 1).collect();
        let groups = chunk_ranges(&lengths, capacity)?;
        let canvas_reserve = if options.sequential {
            lengths.iter().sum()
        } else {
            groups
                .iter()
                .map(|r| lengths[r.clone()].iter().sum::<usize>())
                .max()
                .unwrap_or(0)
        };
        // Close the thought channel before reading answers when no thought was requested.
        // Keep this in prefill so it does not change answer slots or canvas noise.
        let mut suffix = if options.think == 0 {
            self.backend.tokenize(EMPTY_THOUGHT_CHANNEL, false, true)?
        } else {
            Vec::new()
        };
        let thought_reserve = if options.think > 0 {
            options.think
                + self
                    .backend
                    .tokenize(EMPTY_THOUGHT_CHANNEL, false, true)?
                    .len()
        } else {
            suffix.len()
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
            let thought = self.think(&prompt, options.think, seed)?;
            suffix = thought.suffix;
            result.prompt_tokens += thought.input_tokens;
            result.output_tokens = thought.output_tokens;
            result.forward_ms += thought.forward_ms;
        }
        for (group_index, range) in groups.iter().enumerate() {
            let group = &prepared[range.clone()];
            let prompt_length = self.backend.prefill(&prompt, &suffix)?;
            let group_seed = seed.wrapping_add(104729_u64.wrapping_mul(group_index as u64));
            let mut reads = Vec::new();
            for sample in 0..options.samples {
                let sample_seed = group_seed.wrapping_add(7919_u64.wrapping_mul(sample as u64));
                let (slots, canvas, time) =
                    self.read_canvas(group, prompt_length, sample_seed, options.steps)?;
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
            let initial_token = denoise::noise(self.n_vocab, self.mask, &mut rng);
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
        let mut previous = None;
        let mut inverse_temperature = 1.0;
        let start = Instant::now();
        for step in 0..steps {
            self.backend.decode_canvas(
                &canvas,
                prompt_length,
                previous.as_deref(),
                inverse_temperature,
                step + 1 < steps,
            )?;
            if step + 1 == steps {
                for slot in &mut slots {
                    slot.logits = self
                        .backend
                        .logits(slot.canvas_position, &slot.candidate_tokens)?;
                    slot.probabilities = restricted_softmax(&slot.logits)?;
                }
            } else {
                let logits = self.backend.all_logits()?;
                let temperature = 0.4 + 0.4 * (steps - step) as f64 / steps as f64;
                denoise::refine(
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

    fn think(&mut self, prompt: &[PromptPart], budget: usize, seed: u64) -> Result<Thought> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut suffix = self.backend.tokenize("<|channel>thought\n", false, true)?;
        let close = self.backend.tokenize("<channel|>", false, true)?;
        let turn_close = self.backend.tokenize("<turn|>", false, true)?;
        if close.len() != 1 || turn_close.len() != 1 {
            return Err(Error::UnsupportedModel);
        }
        let mut output_tokens = 0;
        let mut input_tokens = 0;
        let mut forward_ms = 0.0;
        while output_tokens < budget {
            let prompt_length = self.backend.prefill(prompt, &suffix)?;
            input_tokens += prompt_length;
            let count = 64.min(self.batch_size).min(budget - output_tokens);
            let mut canvas: Vec<_> = (0..count)
                .map(|_| denoise::noise(self.n_vocab, self.mask, &mut rng))
                .collect();
            let positions: Vec<_> = (0..count).collect();
            let mut previous = None;
            let mut previous_best = Vec::new();
            let mut inverse_temperature = 1.0;
            let start = Instant::now();
            for step in 0..48 {
                self.backend.decode_canvas(
                    &canvas,
                    prompt_length,
                    previous.as_deref(),
                    inverse_temperature,
                    true,
                )?;
                let logits = self.backend.all_logits()?;
                let temperature = 0.4 + 0.4 * (48 - step) as f64 / 48.0;
                let predictions = denoise::refine(
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
            let stop = previous_best
                .iter()
                .position(|t| *t == close[0] || *t == turn_close[0]);
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
        .map(|(text, special, parse)| engine.backend.tokenize(text, special, parse).unwrap().len())
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
}
