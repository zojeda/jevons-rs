//! Restricted-canvas reads: every answer slot is read at once from one canvas forward (or a
//! few refinement steps), optionally after a bounded thought, repeated over samples and
//! chunked when the slots exceed one canvas.

use crate::probability::restricted_softmax;
use crate::{ReadOptions, ReadRequest, ReadResult, SlotRead};
use jevons_core::{Conditioning, DiffusionScheme};
use jevons_core::{Error, ImageInput, Logits, PromptPart, Result};
use jevons_diffusion::{DiffusionEngine, conditioning, masked, uniform};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// The Decision service on a [`DiffusionEngine`].
pub trait Decide {
    /// One read with the default options and no images.
    fn read(&mut self, request: &ReadRequest, seed: u64) -> Result<ReadResult>;

    fn read_with_options(
        &mut self,
        request: &ReadRequest,
        seed: u64,
        options: ReadOptions,
        images: &[ImageInput],
    ) -> Result<ReadResult>;
}

impl Decide for DiffusionEngine {
    fn read(&mut self, request: &ReadRequest, seed: u64) -> Result<ReadResult> {
        self.read_with_options(request, seed, ReadOptions::default(), &[])
    }

    fn read_with_options(
        &mut self,
        request: &ReadRequest,
        seed: u64,
        options: ReadOptions,
        images: &[ImageInput],
    ) -> Result<ReadResult> {
        self.reset_profile();
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
        let capacity = self.canvas_capacity();
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
        let empty_thought = self.tokenize(&self.chat().empty_thought, false, true)?;
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
        if base_length + canvas_reserve + thought_reserve > self.context_size() {
            return Err(Error::InvalidInput(format!(
                "Prompt, thought budget, and canvas need {} tokens; context allows {}",
                base_length + canvas_reserve + thought_reserve,
                self.context_size()
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
            let thought = self.think_with(&prompt, options.think, seed, self.decoding())?;
            suffix = thought.suffix;
            result.prompt_tokens += thought.input_tokens;
            result.output_tokens = thought.output_tokens;
            result.forward_ms += thought.forward_ms;
        }
        for (group_index, range) in groups.iter().enumerate() {
            let group = &prepared[range.clone()];
            let prompt_length = self.model().prefill(&prompt, &suffix)?;
            let group_seed = seed.wrapping_add(104729_u64.wrapping_mul(group_index as u64));
            let mut reads = Vec::new();
            for sample in 0..options.samples {
                let sample_seed = group_seed.wrapping_add(7919_u64.wrapping_mul(sample as u64));
                let (slots, canvas, time) = match self.scheme() {
                    DiffusionScheme::UniformSelfConditioned { .. } => {
                        read_canvas(self, group, prompt_length, sample_seed, options.steps)?
                    }
                    DiffusionScheme::Masked { threshold, .. } => {
                        read_masked(self, group, prompt_length, options.steps, threshold)?
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
}

fn read_canvas(
    engine: &mut DiffusionEngine,
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
        let initial_token = uniform::noise(engine.n_vocab(), engine.mask(), &mut rng);
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
        engine.model().forward_canvas(
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
                slot.logits = engine
                    .model()
                    .candidate_logits(slot.canvas_position, &slot.candidate_tokens)?;
                slot.probabilities = restricted_softmax(&slot.logits)?;
            }
        } else {
            let logits = engine.model().full_logits()?;
            let temperature = 0.4 + 0.4 * (steps - step) as f64 / steps as f64;
            uniform::refine(
                &mut canvas,
                &logits,
                &positions,
                engine.n_vocab() as usize,
                temperature,
                engine.mask(),
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
    engine: &mut DiffusionEngine,
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
        canvas.push(engine.mask());
        slots.push(SlotRead {
            canvas_position: position,
            absolute_position: prompt_length + position,
            initial_token: engine.mask(),
            candidate_tokens: slot.candidates.clone(),
            logits: Vec::new(),
            probabilities: Vec::new(),
        });
    }
    // Trailing masks complete the block, the canvas shape the model was trained on.
    canvas.resize(engine.padded(canvas.len()), engine.mask());
    let mut pending: Vec<usize> = (0..slots.len()).collect();
    let start = Instant::now();
    for step in 0..steps {
        engine.model().forward_canvas(
            &canvas,
            prompt_length,
            Conditioning::None,
            Logits::Candidates,
        )?;
        let mut reads = Vec::with_capacity(pending.len());
        for &i in &pending {
            let logits = engine
                .model()
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

struct PreparedSlot {
    prefix: Vec<i32>,
    candidates: Vec<i32>,
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
    use crate::Slot;
    use jevons_core::TextTokenizer;
    use jevons_diffusion::Decoding;
    use jevons_diffusion::fake::{
        BOS, FakeModel, FakeTokenizer, Log, MASK, fake_engine, masked_model, tokens,
    };

    #[cfg(feature = "models")]
    fn model_test_config() -> jevons_core::ModelConfig {
        jevons_core::ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap())
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

    fn fake_request(slots: usize) -> ReadRequest {
        ReadRequest {
            prompt: " A material. ".into(),
            slots: vec![
                Slot {
                    prefix: "Q: ".into(),
                    candidates: vec!["A".into(), "B".into()],
                };
                slots
            ],
        }
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
        engine.set_context_size(first.prompt_tokens + first.canvas_tokens - 1);
        let error = engine.read(&fake_request(1), 42).unwrap_err();
        assert!(error.to_string().contains("context allows"));
        assert_eq!(log.borrow().prefills.len(), 1);
        assert_eq!(engine.prefill_profile().calls, 0);
        engine.set_context_size(engine.context_size() + 1);
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
        let mut engine = DiffusionEngine::load(&config).unwrap();
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
        let mut config = jevons_core::ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap());
        config.mmproj = Some(std::env::var("DIFFUSION_MMPROJ").unwrap().into());
        let mut engine = DiffusionEngine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag.");
        let before = engine.read(&request, 42).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(224, 224, image::Rgb([255, 0, 0]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let question = ReadRequest {
            prompt: "What color is the image? A = red, B = blue".into(),
            slots: vec![Slot {
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
        let mut engine = DiffusionEngine::load(&config).unwrap();
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
        let context_size = engine.context_size();
        engine.set_context_size(first.prompt_tokens + first.canvas_tokens - 1);
        let error = engine.read(&request, 42).unwrap_err();
        assert!(error.to_string().contains("context allows"));
        assert_eq!(engine.prefill_profile().calls, 0);
        assert_eq!(engine.prefill_profile().processed_tokens, 0);
        engine.set_context_size(engine.context_size() + 1);
        let exact_fit = engine.read(&request, 42).unwrap();
        assert_eq!(exact_fit.prompt_tokens, first.prompt_tokens);
        assert_eq!(
            exact_fit.slots[0].probabilities,
            first.slots[0].probabilities
        );
        engine.set_context_size(context_size);
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
    fn nemotron_engine() -> DiffusionEngine {
        let mut config = jevons_core::ModelConfig::new(std::env::var("NEMOTRON_MODEL").unwrap());
        config.context_size = 4096;
        DiffusionEngine::load(&config).unwrap()
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
        assert_eq!(first.slots[0].initial_token, engine.mask());
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
        assert!(matches!(engine.scheme(), DiffusionScheme::Masked { .. }));
        let budget: usize = std::env::var("THINK_BUDGET")
            .map(|b| b.parse().unwrap())
            .unwrap_or(128);
        let prompts = [
            "What is 15% of 240? Explain the calculation.",
            "Is ground granulated blast furnace slag a supplementary cementitious material? Explain briefly.",
            "Which team should handle this request: \"Compute the least common multiple of 12 and 18\"? The teams are math, coding_agent and writing.",
        ];
        let run = |engine: &mut DiffusionEngine, prompt: &[PromptPart], decoding| {
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
            slots: vec![Slot {
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
