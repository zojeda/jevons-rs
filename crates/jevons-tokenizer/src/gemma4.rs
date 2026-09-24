//! Gemma 4 BPE tokenizer read from GGUF metadata (port of the pinned llama.cpp vocabulary code).
//!
//! This is a line-by-line port of the `gemma4` path of `src/llama-vocab.cpp` in the vendored
//! llama.cpp: vocabulary loading and token attributes (`llama_vocab::impl::load`), special-token
//! partitioning (`tokenizer_st_partition`), the SPM-style BPE session
//! (`llm_tokenizer_bpe_session::tokenize` with `LLAMA_VOCAB_PRE_TYPE_GEMMA4`), and
//! `token_to_piece` with `lstrip = 0` and `special = false`. Output is expected to be
//! token-for-token identical to `llama_tokenize`; the parity test in this module checks that
//! against a dump produced by the linked library.
use jevons_formats::gguf::{Gguf, GgufError, Value};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    ops::Range,
};

const TOKEN_NULL: i32 = -1;

// llama_token_attr bits.
const ATTR_UNDEFINED: u32 = 0;
const ATTR_UNKNOWN: u32 = 1 << 0;
const ATTR_UNUSED: u32 = 1 << 1;
const ATTR_NORMAL: u32 = 1 << 2;
const ATTR_CONTROL: u32 = 1 << 3;
const ATTR_USER_DEFINED: u32 = 1 << 4;
const ATTR_BYTE: u32 = 1 << 5;

/// SentencePiece whitespace marker `▁` (U+2581) used by `escape_whitespaces`.
const SPACE_MARKER: &str = "\u{2581}";

/// Special-token ids read from GGUF, in the order llama.cpp reads them.
const SPECIAL_ID_KEYS: [(&str, Slot); 17] = [
    ("tokenizer.ggml.bos_token_id", Slot::Bos),
    ("tokenizer.ggml.eos_token_id", Slot::Eos),
    ("tokenizer.ggml.eot_token_id", Slot::Eot),
    ("tokenizer.ggml.eom_token_id", Slot::Eom),
    ("tokenizer.ggml.unknown_token_id", Slot::Unk),
    ("tokenizer.ggml.seperator_token_id", Slot::Sep),
    ("tokenizer.ggml.padding_token_id", Slot::Pad),
    ("tokenizer.ggml.mask_token_id", Slot::Mask),
    ("tokenizer.ggml.fim_pre_token_id", Slot::FimPre),
    ("tokenizer.ggml.fim_suf_token_id", Slot::FimSuf),
    ("tokenizer.ggml.fim_mid_token_id", Slot::FimMid),
    ("tokenizer.ggml.fim_pad_token_id", Slot::FimPad),
    ("tokenizer.ggml.fim_rep_token_id", Slot::FimRep),
    ("tokenizer.ggml.fim_sep_token_id", Slot::FimSep),
    ("tokenizer.ggml.prefix_token_id", Slot::FimPre),
    ("tokenizer.ggml.suffix_token_id", Slot::FimSuf),
    ("tokenizer.ggml.middle_token_id", Slot::FimMid),
];

/// Text-detected special tokens (only when the id was not given in metadata); a match gains
/// the CONTROL attribute.
const DETECTED_BY_TEXT: [(Slot, &[&str]); 8] = [
    (
        Slot::Eot,
        &[
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
            "<end_of_turn>",
            "<|endoftext|>",
            "<|end_of_text|>",
            "<EOT>",
            "_<EOT>",
            "[EOT]",
            "<｜end▁of▁sentence｜>",
            "<end_of_utterance>",
        ],
    ),
    (Slot::Eom, &["<|eom_id|>"]),
    (
        Slot::FimPre,
        &[
            "<|fim_prefix|>",
            "<fim-prefix>",
            "<fim_prefix>",
            "<｜fim▁begin｜>",
            "<PRE>",
            "▁<PRE>",
            "<|code_prefix|>",
            "<|prefix|>",
        ],
    ),
    (
        Slot::FimSuf,
        &[
            "<|fim_suffix|>",
            "<fim-suffix>",
            "<fim_suffix>",
            "<｜fim▁hole｜>",
            "<SUF>",
            "▁<SUF>",
            "<|code_suffix|>",
            "<|suffix|>",
        ],
    ),
    (
        Slot::FimMid,
        &[
            "<|fim_middle|>",
            "<fim-middle>",
            "<fim_middle>",
            "<｜fim▁end｜>",
            "<MID>",
            "▁<MID>",
            "<|code_middle|>",
            "<|middle|>",
        ],
    ),
    (
        Slot::FimPad,
        &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"],
    ),
    (
        Slot::FimRep,
        &[
            "<|fim_repo|>",
            "<|repo_name|>",
            "<fim-repo>",
            "<REPO>",
            "<reponame>",
        ],
    ),
    (Slot::FimSep, &["<|file_sep|>"]),
];

/// Token texts that llama.cpp treats as end-of-generation and forces to CONTROL.
const EOG_TEXTS: [&str; 22] = [
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
    "[e~[",
];

/// gpt-oss tokens that llama.cpp always renders (attribute reset to USER_DEFINED).
const ALWAYS_RENDERED: [&str; 4] = ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Bos,
    Eos,
    Eot,
    Eom,
    Unk,
    Sep,
    Pad,
    Mask,
    FimPre,
    FimSuf,
    FimMid,
    FimPad,
    FimRep,
    FimSep,
}

const SLOT_COUNT: usize = 14;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("Unsupported tokenizer model {0:?}; only gemma4 is implemented")]
    UnsupportedModel(String),
    #[error("Invalid tokenizer metadata: {0}")]
    Invalid(String),
}

fn invalid<T>(message: impl Into<String>) -> Result<T, TokenizerError> {
    Err(TokenizerError::Invalid(message.into()))
}

/// Raw tokenizer metadata, independent of the GGUF container (also used by unit tests).
struct VocabSource {
    tokens: Vec<String>,
    token_types: Option<Vec<i64>>,
    merges: Vec<String>,
    special_ids: Vec<(Slot, u64)>,
    add_eos: Option<bool>,
}

/// Gemma 4 tokenizer with the exact semantics of the pinned llama.cpp.
pub struct Tokenizer {
    texts: Vec<Box<str>>,
    attrs: Vec<u32>,
    ids: HashMap<Box<str>, i32>,
    /// Merge rank keyed by the original `"left right"` line; see `merge_rank`.
    ranks: HashMap<Box<str>, u32>,
    /// `cache_special_tokens`: CONTROL, USER_DEFINED, and UNKNOWN tokens, longest text first.
    special: Vec<i32>,
    bos: i32,
    eos: i32,
    mask: i32,
    add_bos: bool,
    add_eos: bool,
}

enum Fragment {
    Text(Range<usize>),
    Token(i32),
}

#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A candidate merge, ordered like `llm_bigram_bpe::comparator`: lowest rank first, then
/// leftmost symbol. The lengths identify the exact text the rank was computed for.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Bigram {
    rank: u32,
    left: usize,
    right: usize,
    left_len: usize,
    right_len: usize,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, TokenizerError> {
        let model = gguf.get("tokenizer.ggml.model")?.as_str().ok_or_else(|| {
            TokenizerError::Invalid("tokenizer.ggml.model is not a string".into())
        })?;
        if model != "gemma4" {
            return Err(TokenizerError::UnsupportedModel(model.into()));
        }
        let strings = |key: &str| -> Result<Vec<String>, TokenizerError> {
            gguf.get(key)?
                .as_array()
                .and_then(|values| {
                    values
                        .iter()
                        .map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .ok_or_else(|| TokenizerError::Invalid(format!("{key} is not a string array")))
        };
        let token_types = match gguf.metadata.get("tokenizer.ggml.token_type") {
            None => None,
            Some(value) => Some(
                value
                    .as_array()
                    .and_then(|values| {
                        values
                            .iter()
                            .map(|v| match v {
                                Value::I32(t) => Some(i64::from(*t)),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| {
                        TokenizerError::Invalid(
                            "tokenizer.ggml.token_type is not an i32 array".into(),
                        )
                    })?,
            ),
        };
        let mut special_ids = Vec::new();
        for (key, slot) in SPECIAL_ID_KEYS {
            if let Some(value) = gguf.metadata.get(key) {
                let id = value
                    .as_u64()
                    .ok_or_else(|| TokenizerError::Invalid(format!("{key} is not an integer")))?;
                special_ids.push((slot, id));
            }
        }
        let flag = |key: &str| -> Result<Option<bool>, TokenizerError> {
            gguf.metadata
                .get(key)
                .map(|v| {
                    v.as_bool()
                        .ok_or_else(|| TokenizerError::Invalid(format!("{key} is not a boolean")))
                })
                .transpose()
        };
        Self::from_source(VocabSource {
            tokens: strings("tokenizer.ggml.tokens")?,
            token_types,
            merges: strings("tokenizer.ggml.merges")?,
            special_ids,
            // tokenizer.ggml.add_bos_token is ignored: llama.cpp forces it on for Gemma 4.
            add_eos: flag("tokenizer.ggml.add_eos_token")?,
        })
    }

    fn from_source(source: VocabSource) -> Result<Self, TokenizerError> {
        let n_tokens = source.tokens.len();
        if n_tokens == 0 || i32::try_from(n_tokens).is_err() {
            return invalid("vocabulary size is out of range");
        }
        if let Some(types) = &source.token_types
            && types.len() < n_tokens
        {
            return invalid("token_type array is shorter than the vocabulary");
        }

        // Merges: `bpe_ranks.emplace` keeps the first rank for a repeated pair. The pair is
        // split at the first space at index >= 1; lookups only ever use space-free halves, so
        // keying by the whole line is equivalent (see `merge_rank`).
        let mut ranks = HashMap::with_capacity(source.merges.len());
        for (rank, line) in source.merges.into_iter().enumerate() {
            let rank = u32::try_from(rank).or_else(|_| invalid("too many merges"))?;
            ranks.entry(line.into_boxed_str()).or_insert(rank);
        }

        let mut texts = Vec::with_capacity(n_tokens);
        let mut attrs = Vec::with_capacity(n_tokens);
        let mut ids = HashMap::with_capacity(n_tokens);
        for (i, mut text) in source.tokens.into_iter().enumerate() {
            if text.is_empty() {
                text = format!("[EMPTY_{i}]");
            }
            let text = text.into_boxed_str();
            if ids.insert(text.clone(), i as i32).is_some() {
                return invalid(format!("duplicate token text {text:?}"));
            }
            texts.push(text);
            attrs.push(match source.token_types.as_ref().map(|types| types[i]) {
                None | Some(1) => ATTR_NORMAL,
                Some(2) => ATTR_UNKNOWN,
                Some(3) => ATTR_CONTROL,
                Some(4) => ATTR_USER_DEFINED,
                Some(5) => ATTR_UNUSED,
                Some(6) => ATTR_BYTE,
                Some(_) => ATTR_UNDEFINED,
            });
        }

        // Special ids: gemma4 defaults are all null; out-of-range metadata keeps the default.
        let mut slots = [TOKEN_NULL; SLOT_COUNT];
        for (slot, id) in source.special_ids {
            if id < n_tokens as u64 {
                slots[slot as usize] = id as i32;
            }
        }
        // Gemma 4 workaround (llama.cpp PR 21500): BOS is always added.
        let add_bos = true;
        let add_eos = source.add_eos.unwrap_or(false);

        let mut tokenizer = Self {
            texts,
            attrs,
            ids,
            ranks,
            special: Vec::new(),
            bos: slots[Slot::Bos as usize],
            eos: slots[Slot::Eos as usize],
            mask: slots[Slot::Mask as usize],
            add_bos,
            add_eos,
        };
        if tokenizer.add_bos && tokenizer.bos == TOKEN_NULL {
            return invalid("the vocabulary adds BOS but has no BOS token");
        }
        if tokenizer.add_eos && tokenizer.eos == TOKEN_NULL {
            return invalid("the vocabulary adds EOS but has no EOS token");
        }
        tokenizer.derive_attributes(&mut slots)?;

        // `cache_special_tokens` uses an unstable std::sort by length; ties are broken by id
        // here. The partition is independent of the order among equal-length tokens unless a
        // proper suffix of one is a prefix of another. That never happens for Gemma 4: every
        // special token starts with '<' and contains no other '<'.
        let mut special: Vec<i32> = (0..n_tokens as i32)
            .filter(|&id| {
                tokenizer.attrs[id as usize] & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN)
                    != 0
            })
            .collect();
        special.sort_by_key(|&id| Reverse(tokenizer.texts[id as usize].len()));
        tokenizer.special = special;
        Ok(tokenizer)
    }

    /// Text-based attribute fixes from `llama_vocab::impl::load`, in the same order.
    fn derive_attributes(&mut self, slots: &mut [i32; SLOT_COUNT]) -> Result<(), TokenizerError> {
        for (slot, candidates) in DETECTED_BY_TEXT {
            if slots[slot as usize] != TOKEN_NULL {
                continue;
            }
            let found: Vec<i32> = candidates.iter().filter_map(|t| self.id(t)).collect();
            match found[..] {
                [] => {}
                [id] => {
                    slots[slot as usize] = id;
                    self.attrs[id as usize] |= ATTR_CONTROL;
                }
                // llama.cpp picks by unordered_map iteration order, which is not portable.
                _ => return invalid(format!("ambiguous {slot:?} tokens {found:?}")),
            }
        }
        for (text, attr) in self.texts.iter().zip(&mut self.attrs) {
            if *attr & ATTR_CONTROL != 0 && text.contains("unused") {
                *attr |= ATTR_UNUSED;
            }
        }
        let mut eog: Vec<i32> = [Slot::FimPad, Slot::FimRep, Slot::FimSep]
            .iter()
            .map(|&slot| slots[slot as usize])
            .filter(|&id| id != TOKEN_NULL)
            .collect();
        for text in EOG_TEXTS {
            if let Some(id) = self.id(text) {
                eog.push(id);
                self.attrs[id as usize] |= ATTR_CONTROL;
            }
        }
        for text in ALWAYS_RENDERED {
            if let Some(id) = self.id(text) {
                self.attrs[id as usize] = ATTR_USER_DEFINED;
            }
        }
        for slot in [Slot::Eos, Slot::Eot, Slot::Eom] {
            if slots[slot as usize] != TOKEN_NULL {
                eog.push(slots[slot as usize]);
            }
        }
        let in_eog =
            |this: &Self, eog: &[i32], text: &str| this.id(text).filter(|id| eog.contains(id));
        let has_call =
            in_eog(self, &eog, "<|call|>").is_some() || in_eog(self, &eog, "<|calls|>").is_some();
        let has_other =
            in_eog(self, &eog, "<|return|>").is_some() || in_eog(self, &eog, "<|flush|>").is_some();
        if let Some(end) = in_eog(self, &eog, "<|end|>")
            && has_call
            && has_other
        {
            eog.retain(|&id| id != end);
            self.attrs[end as usize] = ATTR_USER_DEFINED;
        }
        if in_eog(self, &eog, "<|tool_response>").is_some()
            && let Some(s) = in_eog(self, &eog, "</s>")
        {
            self.attrs[s as usize] = ATTR_NORMAL;
        }
        Ok(())
    }

    fn id(&self, text: &str) -> Option<i32> {
        self.ids.get(text).copied()
    }

    fn text(&self, token: i32) -> &str {
        &self.texts[token as usize]
    }

    pub fn n_vocab(&self) -> usize {
        self.texts.len()
    }

    pub fn bos(&self) -> i32 {
        self.bos
    }

    pub fn eos(&self) -> i32 {
        self.eos
    }

    /// Mask token id, or -1 (`LLAMA_TOKEN_NULL`) when the vocabulary has none.
    pub fn mask(&self) -> i32 {
        self.mask
    }

    pub fn is_control(&self, token: i32) -> bool {
        self.attr(token) & ATTR_CONTROL != 0
    }

    fn attr(&self, token: i32) -> u32 {
        usize::try_from(token)
            .ok()
            .and_then(|i| self.attrs.get(i))
            .copied()
            .unwrap_or(ATTR_UNDEFINED)
    }

    /// Same bytes as `llama_token_to_piece(vocab, token, buf, len, lstrip=0, special=false)`.
    /// Out-of-range ids yield an empty piece.
    pub fn token_to_piece(&self, token: i32) -> Vec<u8> {
        let attr = self.attr(token);
        if attr & (ATTR_UNKNOWN | ATTR_CONTROL) != 0 || attr == ATTR_UNDEFINED {
            return Vec::new();
        }
        let text = self.text(token);
        if attr & ATTR_USER_DEFINED != 0 {
            text.as_bytes().to_vec()
        } else if attr & ATTR_NORMAL != 0 {
            text.replace(SPACE_MARKER, " ").into_bytes()
        } else if attr & ATTR_BYTE != 0 {
            vec![byte_token_value(text)]
        } else {
            Vec::new()
        }
    }

    /// Same tokens as `llama_tokenize(vocab, text, .., add_special, parse_special)`.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        let mut output = Vec::new();
        if add_special && self.add_bos {
            output.push(self.bos);
        }
        if !text.is_empty() {
            for fragment in self.partition(text, parse_special) {
                match fragment {
                    Fragment::Token(id) => output.push(id),
                    // `escape_whitespaces` applies to raw text only, never to special tokens.
                    Fragment::Text(range) => {
                        self.tokenize_bpe(&text[range].replace(' ', SPACE_MARKER), &mut output);
                    }
                }
            }
        }
        if add_special && self.add_eos {
            output.push(self.eos);
        }
        output
    }

    /// `tokenizer_st_partition`: split raw text around special tokens, longest first. CONTROL
    /// and UNKNOWN tokens only participate when `parse_special` is set; USER_DEFINED always do.
    fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut fragments = vec![Fragment::Text(0..text.len())];
        for &id in &self.special {
            if !parse_special && self.attrs[id as usize] & (ATTR_CONTROL | ATTR_UNKNOWN) != 0 {
                continue;
            }
            let needle = self.text(id);
            if !fragments
                .iter()
                .any(|f| matches!(f, Fragment::Text(r) if text[r.clone()].contains(needle)))
            {
                continue;
            }
            let mut next = Vec::with_capacity(fragments.len() + 2);
            for fragment in fragments {
                let Fragment::Text(range) = fragment else {
                    next.push(fragment);
                    continue;
                };
                let mut start = range.start;
                while let Some(found) = text[start..range.end].find(needle) {
                    let at = start + found;
                    if at > start {
                        next.push(Fragment::Text(start..at));
                    }
                    next.push(Fragment::Token(id));
                    start = at + needle.len();
                }
                if start < range.end {
                    next.push(Fragment::Text(start..range.end));
                }
            }
            fragments = next;
        }
        fragments
    }

    /// `llm_tokenizer_bpe_session::tokenize` for `LLAMA_VOCAB_PRE_TYPE_GEMMA4`.
    fn tokenize_bpe(&self, text: &str, output: &mut Vec<i32>) {
        let mut symbols = Vec::new();
        let mut queue = BinaryHeap::new();
        let mut key = String::new();
        for word in newline_runs(text) {
            symbols.clear();
            queue.clear();
            if word.bytes().all(|b| b == b'\n') && self.ids.contains_key(word) {
                // Gemma 4 newline runs map to a single token (llama.cpp PR 21343).
                symbols.push(Symbol {
                    start: 0,
                    len: word.len(),
                    prev: None,
                    next: None,
                });
            } else {
                let mut chars = word.char_indices().peekable();
                while let Some((start, c)) = chars.next() {
                    let index = symbols.len();
                    symbols.push(Symbol {
                        start,
                        len: c.len_utf8(),
                        prev: index.checked_sub(1),
                        next: chars.peek().map(|_| index + 1),
                    });
                }
            }
            for right in 1..symbols.len() {
                self.push_bigram(word, &symbols, right - 1, right, &mut key, &mut queue);
            }
            while let Some(Reverse(bigram)) = queue.pop() {
                let (left, right) = (symbols[bigram.left], symbols[bigram.right]);
                // Skip outdated bigrams: a symbol grew or was absorbed since the push.
                if left.len == 0
                    || right.len == 0
                    || left.len != bigram.left_len
                    || right.len != bigram.right_len
                {
                    continue;
                }
                symbols[bigram.left].len += right.len;
                symbols[bigram.right].len = 0;
                symbols[bigram.left].next = right.next;
                if let Some(next) = right.next {
                    symbols[next].prev = Some(bigram.left);
                }
                if let Some(prev) = left.prev {
                    self.push_bigram(word, &symbols, prev, bigram.left, &mut key, &mut queue);
                }
                if let Some(next) = right.next {
                    self.push_bigram(word, &symbols, bigram.left, next, &mut key, &mut queue);
                }
            }
            for symbol in symbols.iter().filter(|s| s.len > 0) {
                let piece = &word[symbol.start..symbol.start + symbol.len];
                match self.id(piece) {
                    Some(id) => output.push(id),
                    None => {
                        // Byte fallback with SPM-style `<0xXX>` tokens; missing bytes are dropped.
                        for byte in piece.bytes() {
                            if let Some(id) = self.id(&format!("<0x{byte:02X}>")) {
                                output.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    fn push_bigram(
        &self,
        word: &str,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        key: &mut String,
        queue: &mut BinaryHeap<Reverse<Bigram>>,
    ) {
        let (l, r) = (symbols[left], symbols[right]);
        let left_text = &word[l.start..l.start + l.len];
        let right_text = &word[r.start..r.start + r.len];
        if let Some(rank) = self.merge_rank(left_text, right_text, key) {
            queue.push(Reverse(Bigram {
                rank,
                left,
                right,
                left_len: l.len,
                right_len: r.len,
            }));
        }
    }

    /// `find_bpe_rank`. Both halves are non-empty and space-free (spaces are escaped), so the
    /// pair `(left, right)` was produced by splitting exactly the merge line `"left right"`.
    fn merge_rank(&self, left: &str, right: &str, key: &mut String) -> Option<u32> {
        key.clear();
        key.push_str(left);
        key.push(' ');
        key.push_str(right);
        self.ranks.get(key.as_str()).copied()
    }
}

/// `unicode_regex_split_custom_newlines` for `[^\n]+|[\n]+`: alternating runs of newlines and
/// non-newlines (byte-exact for valid UTF-8, since `\n` never occurs inside a multibyte char).
fn newline_runs(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= bytes.len() {
            return None;
        }
        let newline = bytes[start] == b'\n';
        let end = bytes[start..]
            .iter()
            .position(|&b| (b == b'\n') != newline)
            .map_or(bytes.len(), |n| start + n);
        let run = &text[start..end];
        start = end;
        Some(run)
    })
}

/// `token_to_byte`: `strtol(text.substr(3, 2), NULL, 16)` for `<0xXX>` byte tokens.
fn byte_token_value(text: &str) -> u8 {
    let digits = text.as_bytes().get(3..).unwrap_or_default();
    let mut value = 0u8;
    for &digit in digits.iter().take(2) {
        match (digit as char).to_digit(16) {
            Some(d) => value = value.wrapping_mul(16).wrapping_add(d as u8),
            None => break,
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny vocabulary: ids 0..=4 are control/special, then byte tokens, then normal pieces.
    fn synthetic() -> Tokenizer {
        let mut tokens: Vec<(String, i64)> = vec![
            ("<pad>".into(), 3),
            ("<eos>".into(), 3),
            ("<bos>".into(), 3),
            ("<unk>".into(), 3),
            ("<mask>".into(), 3),
            ("<|turn>".into(), 3),
            ("<turn|>".into(), 3),
            ("<|tool_response>".into(), 4),
            ("<|\"|>".into(), 4),
        ];
        for byte in 0..=255u8 {
            tokens.push((format!("<0x{byte:02X}>"), 6));
        }
        for piece in [
            "a", "b", "c", "ab", "bc", "abc", "▁", "▁a", "▁ab", "\n", "\n\n", "x", "é", "</s>",
        ] {
            tokens.push((piece.into(), 1));
        }
        let merges = ["b c", "a b", "▁ a", "▁a b", "a bc", "\n \n"]
            .map(String::from)
            .to_vec();
        let (tokens, types): (Vec<_>, Vec<_>) = tokens.into_iter().unzip();
        Tokenizer::from_source(VocabSource {
            tokens,
            token_types: Some(types),
            merges,
            special_ids: vec![(Slot::Bos, 2), (Slot::Eos, 1), (Slot::Mask, 4)],
            add_eos: None,
        })
        .unwrap()
    }

    fn pieces(tok: &Tokenizer, ids: &[i32]) -> Vec<String> {
        ids.iter().map(|&id| tok.text(id).to_owned()).collect()
    }

    #[test]
    fn lower_rank_merges_win_over_leftmost_pairs() {
        let tok = synthetic();
        // "b c" (rank 0) beats "a b" (rank 1), so "abc" is reached through "a"+"bc".
        assert_eq!(pieces(&tok, &tok.tokenize("abc", false, false)), ["abc"]);
        // Without "a bc" reachable, "bcab" becomes "bc" then "ab".
        assert_eq!(
            pieces(&tok, &tok.tokenize("bcab", false, false)),
            ["bc", "ab"]
        );
    }

    #[test]
    fn spaces_are_escaped_and_merged_as_markers() {
        let tok = synthetic();
        assert_eq!(pieces(&tok, &tok.tokenize(" a", false, false)), ["▁a"]);
        // "a b" (rank 1) merges before "▁ a" (rank 2), and "▁ ab" is not a merge.
        assert_eq!(
            pieces(&tok, &tok.tokenize(" ab", false, false)),
            ["▁", "ab"]
        );
        assert_eq!(
            pieces(&tok, &tok.tokenize("a  a", false, false)),
            ["a", "▁", "▁a"]
        );
        let ab = tok.id("▁ab").unwrap();
        assert_eq!(tok.token_to_piece(ab), b" ab");
    }

    #[test]
    fn newline_runs_use_whole_tokens_or_merge_per_character() {
        let tok = synthetic();
        assert_eq!(
            pieces(&tok, &tok.tokenize("a\n\nb", false, false)),
            ["a", "\n\n", "b"]
        );
        // "\n\n\n" is not a token: characters merge through "\n \n" into "\n\n" + "\n".
        assert_eq!(
            pieces(&tok, &tok.tokenize("\n\n\n", false, false)),
            ["\n\n", "\n"]
        );
    }

    #[test]
    fn unknown_characters_fall_back_to_byte_tokens() {
        let tok = synthetic();
        let ids = tok.tokenize("aé€", false, false);
        let expected = ["a", "é", "<0xE2>", "<0x82>", "<0xAC>"];
        assert_eq!(pieces(&tok, &ids), expected);
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| tok.token_to_piece(id)).collect();
        assert_eq!(bytes, "aé€".as_bytes());
    }

    #[test]
    fn control_tokens_are_parsed_only_when_parse_special_is_set() {
        let tok = synthetic();
        let parsed = tok.tokenize("<|turn>ab<turn|>", true, true);
        assert_eq!(pieces(&tok, &parsed), ["<bos>", "<|turn>", "ab", "<turn|>"]);
        let literal = tok.tokenize("<|turn>", false, false);
        assert!(literal.iter().all(|&id| !tok.is_control(id)));
        assert_eq!(literal.len(), "<|turn>".len());
    }

    #[test]
    fn user_defined_tokens_are_always_partitioned_and_spaces_stay_raw() {
        let tok = synthetic();
        let ids = tok.tokenize("a <|\"|> a", false, false);
        assert_eq!(pieces(&tok, &ids), ["a", "▁", "<|\"|>", "▁a"]);
        // `<|tool_response>` becomes CONTROL via the EOG list, so it needs parse_special.
        assert!(tok.is_control(tok.id("<|tool_response>").unwrap()));
        assert_eq!(tok.tokenize("<|tool_response>", false, true), [7]);
        assert_ne!(tok.tokenize("<|tool_response>", false, false), [7]);
    }

    #[test]
    fn gemma4_always_adds_bos_but_not_eos() {
        let tok = synthetic();
        assert_eq!(tok.tokenize("", true, false), [2]);
        assert!(tok.tokenize("", false, false).is_empty());
        assert_eq!(tok.tokenize("a", true, true), [2, tok.id("a").unwrap()]);
    }

    #[test]
    fn pieces_hide_control_tokens_and_decode_bytes() {
        let tok = synthetic();
        assert!(tok.token_to_piece(1).is_empty());
        assert!(tok.token_to_piece(7).is_empty());
        assert_eq!(tok.token_to_piece(8), b"<|\"|>");
        assert_eq!(tok.token_to_piece(tok.id("<0x0A>").unwrap()), b"\n");
        // `</s>` is reset to NORMAL because `<|tool_response>` is an EOG token.
        assert_eq!(tok.token_to_piece(tok.id("</s>").unwrap()), b"</s>");
        assert!(tok.token_to_piece(-1).is_empty());
        assert!(tok.token_to_piece(tok.n_vocab() as i32).is_empty());
    }

    /// Compares against a dump of llama.cpp's tokenizer output. The dump writer was the
    /// `writes_tokenizer_reference_dump` test of the removed llama.cpp backend (see git history);
    /// parity was verified on all 262,144 token pieces and 269,290 strings before its removal.
    #[test]
    #[ignore = "Requires DIFFUSION_MODEL and TOKENIZER_REFERENCE (a llama.cpp dump)"]
    fn tokenizer_matches_llama_reference() {
        let model = std::env::var("DIFFUSION_MODEL").expect("DIFFUSION_MODEL");
        let dump = std::env::var("TOKENIZER_REFERENCE").expect("TOKENIZER_REFERENCE");
        let started = std::time::Instant::now();
        let gguf = Gguf::open(&model).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        eprintln!("loaded tokenizer in {:?}", started.elapsed());
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&dump).unwrap()).unwrap();
        let array = |key: &str| reference[key].as_array().unwrap().as_slice();
        let (pieces_hex, control, cases) = (array("pieces_hex"), array("control"), array("cases"));

        assert_eq!(pieces_hex.len(), tok.n_vocab());
        assert_eq!(control.len(), tok.n_vocab());
        let mut piece_mismatches = Vec::new();
        for (id, (hex, control)) in pieces_hex.iter().zip(control).enumerate() {
            let (id, hex, control) = (id as i32, hex.as_str().unwrap(), control.as_bool().unwrap());
            let piece: String = tok
                .token_to_piece(id)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if piece != hex || tok.is_control(id) != control {
                piece_mismatches.push((id, hex.to_owned(), piece, control));
            }
        }

        let started = std::time::Instant::now();
        let mut case_mismatches = Vec::new();
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<i32> = case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_i64().unwrap() as i32)
                .collect();
            let add_special = case["add_special"].as_bool().unwrap();
            let parse_special = case["parse_special"].as_bool().unwrap();
            let tokens = tok.tokenize(text, add_special, parse_special);
            if tokens != expected {
                case_mismatches.push((
                    text.to_owned(),
                    add_special,
                    parse_special,
                    expected,
                    tokens,
                ));
            }
        }
        eprintln!(
            "compared {} strings in {:?} and {} pieces: {} string and {} piece mismatches",
            cases.len(),
            started.elapsed(),
            tok.n_vocab(),
            case_mismatches.len(),
            piece_mismatches.len()
        );
        for mismatch in case_mismatches.iter().take(10) {
            eprintln!("string mismatch: {mismatch:?}");
        }
        for mismatch in piece_mismatches.iter().take(10) {
            eprintln!("piece mismatch: {mismatch:?}");
        }
        assert!(case_mismatches.is_empty() && piece_mismatches.is_empty());
    }
}
