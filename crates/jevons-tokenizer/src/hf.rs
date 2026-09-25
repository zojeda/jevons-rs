//! Hugging Face `tokenizer.json` tokenizers (BPE/ByteLevel, such as Tekken).
//!
//! Chat markers are added tokens, and some of them are not flagged special (Nemotron's `<think>`
//! and `</think>`), so Hugging Face encodes them even in plain text. User text must never produce
//! control tokens, so literal encoding runs a second tokenizer built from the same file with no
//! added tokens, and rejects any result that still contains one.
use jevons_core::{Error, Result, TextTokenizer};
use std::collections::HashSet;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct HfTokenizer {
    /// Parses added tokens: used for chat markers.
    framing: Tokenizer,
    /// Byte-level BPE only: used for user text.
    literal: Tokenizer,
    added: HashSet<u32>,
    bos: Option<i32>,
    vocab: usize,
}

fn load_error(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::UnsupportedModel(format!("tokenizer {}: {error}", path.display()))
}

impl HfTokenizer {
    /// Loads `tokenizer.json`. `bos` is the beginning-of-sequence token, if the model uses one.
    pub fn from_file(path: &Path, bos: Option<i32>) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|_| Error::ModelLoad)?;
        let mut json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| load_error(path, e))?;
        let framing = Tokenizer::from_bytes(text.as_bytes()).map_err(|e| load_error(path, e))?;
        let added: HashSet<u32> = json["added_tokens"]
            .as_array()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|t| t["id"].as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect()
            })
            .unwrap_or_default();
        json["added_tokens"] = serde_json::Value::Array(Vec::new());
        let literal =
            Tokenizer::from_bytes(json.to_string().as_bytes()).map_err(|e| load_error(path, e))?;
        let vocab = framing.get_vocab_size(true);
        Ok(Self {
            framing,
            literal,
            added,
            bos,
            vocab,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.vocab
    }

    /// Whether `token` is an added (control or chat-marker) token.
    pub fn is_added(&self, token: i32) -> bool {
        u32::try_from(token).is_ok_and(|t| self.added.contains(&t))
    }

    /// The single token `text` encodes to with markers parsed, if it is one.
    pub fn single_token(&self, text: &str) -> Option<i32> {
        match self.tokenize(text, false, true).ok()?[..] {
            [token] => Some(token),
            _ => None,
        }
    }
}

impl TextTokenizer for HfTokenizer {
    fn tokenize(&self, text: &str, bos: bool, special: bool) -> Result<Vec<i32>> {
        let tokenizer = if special {
            &self.framing
        } else {
            &self.literal
        };
        let encoding = tokenizer
            .encode_fast(text, false)
            .map_err(|e| Error::InvalidInput(format!("Cannot tokenize text: {e}")))?;
        let ids = encoding.get_ids();
        if !special && ids.iter().any(|id| self.added.contains(id)) {
            return Err(Error::InvalidInput(
                "Text encodes to a control token".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(ids.len() + 1);
        if bos && let Some(bos) = self.bos {
            tokens.push(bos);
        }
        tokens.extend(ids.iter().map(|&id| id as i32));
        Ok(tokens)
    }

    fn code_piece(&self, token: i32) -> Option<String> {
        let id = u32::try_from(token).ok()?;
        if id as usize >= self.vocab || self.added.contains(&id) {
            return None;
        }
        let piece = self.literal.decode(&[id], false).ok()?;
        let piece = piece.strip_prefix(' ').unwrap_or(&piece);
        ((1..=16).contains(&piece.len()) && piece.bytes().all(|b| b.is_ascii_alphanumeric()))
            .then(|| piece.to_string())
    }

    fn decode(&self, tokens: &[i32]) -> Result<String> {
        let ids = tokens
            .iter()
            .map(|&t| {
                u32::try_from(t)
                    .ok()
                    .filter(|&id| (id as usize) < self.vocab)
            })
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(|| Error::InvalidInput("token id out of range".into()))?;
        self.framing
            .decode(&ids, true)
            .map_err(|e| Error::Backend(format!("Cannot decode tokens: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal ByteLevel BPE with one special and one non-special added marker.
    fn tokenizer() -> HfTokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": {"<s>": 0, "</think>": 1, "<": 2, "/": 3, "t": 4, "h": 5, "i": 6, "n": 7, "k": 8, ">": 9,
                          "A": 10, "B": 11, "Ġ": 12, "th": 13, "ink": 14, "in": 15},
                "merges": ["t h", "i n", "in k"]
            }
        });
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("jevons-hf-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json.to_string()).unwrap();
        let tokenizer = HfTokenizer::from_file(&path, Some(0)).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        tokenizer
    }

    #[test]
    fn markers_parse_only_in_framing_and_user_text_never_yields_them() {
        let tok = tokenizer();
        assert_eq!(tok.tokenize("</think>", false, true).unwrap(), [1]);
        assert_eq!(tok.single_token("</think>"), Some(1));
        let literal = tok.tokenize("</think>", false, false).unwrap();
        assert_eq!(literal, [2, 3, 13, 14, 9]);
        assert!(!literal.iter().any(|&t| tok.is_added(t)));
        assert_eq!(tok.tokenize("A", true, false).unwrap(), [0, 10]);
    }

    #[test]
    #[ignore = "Requires NEMOTRON_MODEL (a Nemotron-Labs-Diffusion checkpoint directory)"]
    fn nemotron_chat_markers_are_single_tokens_and_literal_text_stays_plain() {
        let dir = std::path::PathBuf::from(std::env::var("NEMOTRON_MODEL").unwrap());
        let tok = HfTokenizer::from_file(&dir.join("tokenizer.json"), None).unwrap();
        assert_eq!(tok.n_vocab(), 131_073);
        for (marker, id) in [("<|im_start|>", 10), ("<|im_end|>", 11), ("</think>", 13)] {
            assert_eq!(tok.single_token(marker), Some(id), "{marker}");
        }
        let chat = tok
            .tokenize("<|im_start|>user\nHi<|im_end|>", false, true)
            .unwrap();
        assert_eq!((chat[0], *chat.last().unwrap()), (10, 11));
        for text in ["</think> <|im_start|> [INST] <s> <SPECIAL_100>", "|<MASK>|"] {
            let literal = tok.tokenize(text, false, false).unwrap();
            assert!(
                !literal.iter().any(|&t| tok.is_added(t)),
                "{text}: {literal:?}"
            );
        }
        let codes = (0..tok.n_vocab() as i32)
            .filter_map(|t| {
                tok.code_piece(t)
                    .filter(|c| tok.tokenize(c, false, false).unwrap() == [t])
            })
            .count();
        assert!(codes >= 128, "{codes}");
    }

    #[test]
    fn code_pieces_exclude_added_tokens_and_non_alphanumeric_text() {
        let tok = tokenizer();
        assert_eq!(tok.code_piece(10).as_deref(), Some("A"));
        assert_eq!(tok.code_piece(14).as_deref(), Some("ink"));
        assert_eq!(tok.code_piece(1), None);
        assert_eq!(tok.code_piece(9), None);
        assert_eq!(tok.code_piece(99), None);
    }
}
