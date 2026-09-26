//! Free-form answers: chat framing, an optional bounded thought, and the answer streamed as it
//! is decided, cut at stop sequences that are never streamed in part.

use crate::{FinishReason, Generation, GenerationPrompt, GenerationRequest, Role};
use jevons_core::{Error, PromptPart, Result, TextTokenizer};
use jevons_diffusion::DiffusionEngine;

/// The Generative service on a [`DiffusionEngine`].
pub trait Generate {
    /// Generates a free-form answer (the OpenAI-compatible endpoints). `on_text` receives the
    /// answer text in order as it is decided and returns false to stop, for example when the
    /// client has gone. Text that could still become a stop sequence is held back until it
    /// cannot.
    fn generate(
        &mut self,
        request: &GenerationRequest,
        seed: u64,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Generation>;
}

impl Generate for DiffusionEngine {
    fn generate(
        &mut self,
        request: &GenerationRequest,
        seed: u64,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Generation> {
        request.validate()?;
        self.reset_profile();
        let (prompt, framing) = match &request.prompt {
            GenerationPrompt::Chat(messages) => {
                let mut tokens = if self.chat().bos {
                    self.tokenize("", true, false)?
                } else {
                    Vec::new()
                };
                for message in messages {
                    let (open, prefix) = match message.role {
                        Role::System => (&self.chat().system_open, ""),
                        Role::User => (&self.chat().user_open, ""),
                        Role::Assistant => (
                            &self.chat().assistant_open,
                            self.chat().history_prefix.as_str(),
                        ),
                    };
                    tokens.extend(self.tokenize(open, false, true)?);
                    tokens.extend(self.tokenize(prefix, false, true)?);
                    tokens.extend(self.tokenize(&message.text, false, false)?);
                    tokens.extend(self.tokenize(&self.chat().turn_close, false, true)?);
                }
                tokens.extend(self.tokenize(&self.chat().assistant_open, false, true)?);
                let framing = if request.think > 0 {
                    Vec::new()
                } else {
                    self.tokenize(&self.chat().empty_thought, false, true)?
                };
                (tokens, framing)
            }
            GenerationPrompt::Text(text) => {
                (self.tokenize(text, self.chat().bos, false)?, Vec::new())
            }
        };
        if prompt.is_empty() {
            return Err(Error::InvalidInput("The prompt is empty".into()));
        }
        let thought_reserve = if request.think > 0 {
            request.think + self.tokenize(&self.chat().thought_open, false, true)?.len() + 1
        } else {
            framing.len()
        };
        let available = self
            .context_size()
            .saturating_sub(prompt.len() + thought_reserve);
        let max_tokens = request
            .max_tokens
            .unwrap_or(DEFAULT_MAX_TOKENS.min(available));
        if max_tokens == 0 || max_tokens > available {
            return Err(Error::InvalidInput(format!(
                "The prompt needs {} tokens and the answer up to {max_tokens}; the context allows {}",
                prompt.len() + thought_reserve,
                self.context_size()
            )));
        }
        let prompt_tokens = prompt.len();
        let prompt = [PromptPart::Text(prompt)];
        let (start, reasoning_tokens) = if request.think > 0 {
            let thought = self.think_with(&prompt, request.think, seed, self.decoding())?;
            (thought.suffix, thought.output_tokens)
        } else {
            (framing, 0)
        };
        let stops = self.marker_tokens(&self.chat().answer_stops)?;
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
            self.decoding(),
            &mut sink,
        )?;
        if let Some(error) = answer.error.take() {
            return Err(error);
        }
        if !answer.stopped && !answer.gone {
            let text = self.tokenizer().decode(&answer.tokens)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use jevons_diffusion::Decoding;
    use jevons_diffusion::fake::{Log, fake_engine, masked_model, tokens};

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
}
