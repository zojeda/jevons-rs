//! Free-form text generation, as served by the OpenAI-compatible endpoints.

/// Who wrote a conversation turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Instructions (OpenAI `system` and `developer`).
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerationPrompt {
    /// A conversation, framed with the model's chat markers; the answer opens a model turn.
    Chat(Vec<Message>),
    /// Raw text continued as is (the legacy Completions API). It never parses chat markers.
    Text(String),
}

/// At most this many stop sequences, as in the OpenAI APIs.
pub const MAX_STOP_SEQUENCES: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationRequest {
    pub prompt: GenerationPrompt,
    /// Largest number of answer tokens; `None` allows the rest of the context, up to a cap.
    pub max_tokens: Option<usize>,
    /// Thought budget before a chat answer (0 answers directly).
    pub think: usize,
    /// Text that ends the answer when generated; it is not returned.
    pub stop: Vec<String>,
}

impl GenerationRequest {
    pub fn validate(&self) -> crate::Result<()> {
        let invalid = |message: &str| Err(crate::Error::InvalidInput(message.into()));
        match &self.prompt {
            GenerationPrompt::Chat(messages) if messages.is_empty() => {
                return invalid("At least one message is required");
            }
            GenerationPrompt::Text(_) if self.think > 0 => {
                return invalid("Text completions cannot think");
            }
            _ => {}
        }
        if self.max_tokens == Some(0) {
            return invalid("max_tokens must be positive");
        }
        if self.think > 4096 {
            return invalid("The thought budget is at most 4096 tokens");
        }
        if self.stop.len() > MAX_STOP_SEQUENCES || self.stop.iter().any(String::is_empty) {
            return invalid("Use at most 4 nonempty stop sequences");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended its turn, or a stop sequence was generated.
    Stop,
    /// The answer reached `max_tokens`.
    Length,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub text: String,
    /// Prompt tokens, including chat framing.
    pub prompt_tokens: usize,
    /// Generated tokens, thought included.
    pub completion_tokens: usize,
    /// Thought tokens among `completion_tokens`.
    pub reasoning_tokens: usize,
    pub finish: FinishReason,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_requests_reject_empty_conversations_and_bad_limits() {
        let chat = |messages| GenerationRequest {
            prompt: GenerationPrompt::Chat(messages),
            max_tokens: Some(8),
            think: 0,
            stop: Vec::new(),
        };
        let hello = vec![Message {
            role: Role::User,
            text: "Hi".into(),
        }];
        assert!(chat(hello.clone()).validate().is_ok());
        assert!(chat(Vec::new()).validate().is_err());
        let mut request = chat(hello);
        request.max_tokens = Some(0);
        assert!(request.validate().is_err());
        request.max_tokens = None;
        request.stop = vec!["a".into(); 5];
        assert!(request.validate().is_err());
        request.stop = vec![String::new()];
        assert!(request.validate().is_err());
        let text = GenerationRequest {
            prompt: GenerationPrompt::Text("Once".into()),
            max_tokens: None,
            think: 16,
            stop: Vec::new(),
        };
        assert!(text.validate().is_err());
    }
}
