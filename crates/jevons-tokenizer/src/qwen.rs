//! Qwen's byte-level BPE from a checkpoint's `vocab.json` and `merges.txt`, for checkpoints
//! that ship no `tokenizer.json` (Qwen3-TTS), cut into pieces as the `qwen-tts` package does.
//!
//! It encodes plain text only: no added token is parsed, so text never yields a chat marker.
//! The caller writes the markers by id.
use jevons_core::{Error, Result};
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{SplitDelimiterBehavior, Tokenizer};

/// How the `qwen-tts` package cuts text into the pieces BPE merges inside. It loads
/// `Qwen2Tokenizer` with `transformers`' `fix_mistral_regex`, which replaces Qwen's own pattern
/// with this one: a word with the character before it, ending where an upper-case letter
/// follows a lower-case one; one digit; a run of punctuation with the slashes and line breaks
/// after it; line breaks; other whitespace. "GitHub" is two pieces here and one in Qwen's
/// pattern, so the ids differ, and the port follows the package it is checked against.
const PIECES: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct QwenBpe {
    tokenizer: Tokenizer,
}

fn load_error(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::UnsupportedModel(format!("tokenizer {}: {error}", path.display()))
}

impl QwenBpe {
    /// Loads `vocab.json` and `merges.txt`.
    pub fn from_files(vocab: &Path, merges: &Path) -> Result<Self> {
        let (Some(vocab_path), Some(merges_path)) = (vocab.to_str(), merges.to_str()) else {
            return Err(Error::ModelLoad);
        };
        if !vocab.is_file() || !merges.is_file() {
            return Err(Error::ModelLoad);
        }
        let model = BPE::from_file(vocab_path, merges_path)
            .build()
            .map_err(|e| load_error(vocab, e))?;
        let mut tokenizer = Tokenizer::new(model);
        let split = Split::new(
            SplitPattern::Regex(PIECES.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| load_error(vocab, e))?;
        tokenizer
            .with_normalizer(Some(NFC))
            .map_err(|e| load_error(vocab, e))?
            .with_pre_tokenizer(Some(Sequence::new(vec![
                split.into(),
                ByteLevel::new(false, false, false).into(),
            ])));
        Ok(Self { tokenizer })
    }

    /// The token ids of `text`, read as plain text.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self
            .tokenizer
            .encode_fast(text, false)
            .map_err(|e| Error::InvalidInput(format!("Cannot tokenize text: {e}")))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary of single bytes and three merges, in the files' formats.
    fn tiny(test: &str) -> QwenBpe {
        let dir =
            std::env::temp_dir().join(format!("jevons-qwen-bpe-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pieces = [
            "h", "o", "l", "a", "Ġ", "<", "|", ">", "ho", "la", "Ġho", "Ã", "±", "H", "laH",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| (piece.to_string(), id.into()))
            .collect();
        std::fs::write(
            dir.join("vocab.json"),
            serde_json::to_string(&vocab).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("merges.txt"), "h o\nl a\nĠ ho\nla H\n").unwrap();
        let bpe = QwenBpe::from_files(&dir.join("vocab.json"), &dir.join("merges.txt")).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        bpe
    }

    #[test]
    fn text_merges_inside_words_and_a_word_keeps_its_leading_space() {
        let bpe = tiny("merges");
        assert_eq!(bpe.encode("hola hola").unwrap(), vec![8, 9, 10, 9]);
    }

    #[test]
    fn a_word_ends_where_an_upper_case_letter_follows_a_lower_case_one() {
        let bpe = tiny("camel");
        // "la" and "H" merge inside one piece; "holaHola" is two.
        assert_eq!(bpe.encode("holaHola").unwrap(), vec![8, 9, 13, 1, 9]);
        assert_eq!(bpe.encode("Hola").unwrap(), vec![13, 1, 9]);
    }

    #[test]
    fn a_chat_marker_in_text_stays_plain_text() {
        let bpe = tiny("marker");
        // "<|" and "|>" are punctuation runs, never one marker token.
        assert_eq!(bpe.encode("<|hola|>").unwrap(), vec![5, 6, 8, 9, 6, 7]);
    }

    #[test]
    fn decomposed_letters_are_composed_before_encoding() {
        let bpe = tiny("composed");
        // "n" + U+0303 composes to "ñ", whose two UTF-8 bytes are in the vocabulary.
        assert_eq!(bpe.encode("n\u{303}").unwrap(), vec![11, 12]);
    }

    #[test]
    fn missing_files_fail_as_a_model_that_does_not_load() {
        let missing = Path::new("/nonexistent/vocab.json");
        assert!(matches!(
            QwenBpe::from_files(missing, missing),
            Err(Error::ModelLoad)
        ));
    }
}
