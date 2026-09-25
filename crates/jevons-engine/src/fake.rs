//! A scripted in-memory model for engine tests without a GPU or model files.
use jevons_core::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Error, ImageInput, Logits,
    ModelInfo, PrefillProfile, PromptPart, Result, TextTokenizer,
};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) const BOS: i32 = 1;
pub(crate) const MASK: i32 = 4;
const VOCAB: i32 = 512;
/// Markers parsed only with special tokens enabled.
const MARKERS: [(&str, i32); 6] = [
    ("<user>", 10),
    ("<model>", 11),
    ("<think>", 12),
    ("</think>", 13),
    ("<end>", 14),
    ("<nothought>", 15),
];
/// Two-character words tokenized as one token: "w0".."w99" are tokens 300..400.
const WORDS: i32 = 300;
/// With `space_joins`, a space joins the next word: " w0".." w99" are tokens 400..500 and a
/// space before an ASCII alphanumeric is token 150 + its index in `ALNUM`.
const SPACED_WORDS: i32 = 400;
const SPACED_CHARS: i32 = 150;
const ALNUM: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Everything the engine asked of the model, for assertions.
#[derive(Default)]
pub(crate) struct Log {
    pub prefills: Vec<Vec<i32>>,
    pub canvases: Vec<Vec<i32>>,
    pub conditioned: Vec<bool>,
    /// Rows requested by each causal prediction, in call order.
    pub predicted: Vec<usize>,
}

pub(crate) struct FakeTokenizer {
    pub space_joins: bool,
}

impl TextTokenizer for FakeTokenizer {
    fn tokenize(&self, text: &str, bos: bool, special: bool) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        if bos {
            tokens.push(BOS);
        }
        let mut rest = text;
        'outer: while let Some(ch) = rest.chars().next() {
            if special {
                for (marker, token) in MARKERS {
                    if let Some(tail) = rest.strip_prefix(marker) {
                        tokens.push(token);
                        rest = tail;
                        continue 'outer;
                    }
                }
            }
            let bytes = rest.as_bytes();
            let spaced = self.space_joins && bytes.len() >= 2 && bytes[0] == b' ';
            let word = &bytes[usize::from(spaced)..];
            if word.len() >= 2 && word[0] == b'w' && word[1].is_ascii_digit() {
                let digits = word[1..]
                    .iter()
                    .take(2)
                    .take_while(|b| b.is_ascii_digit())
                    .count();
                let start = usize::from(spaced) + 1;
                let n: i32 = rest[start..start + digits].parse().unwrap();
                tokens.push(if spaced { SPACED_WORDS } else { WORDS } + n);
                rest = &rest[start + digits..];
                continue;
            }
            if spaced && let Some(i) = ALNUM.find(char::from(bytes[1])) {
                tokens.push(SPACED_CHARS + i as i32);
                rest = &rest[2..];
                continue;
            }
            tokens.push(i32::from(u8::try_from(ch).unwrap_or(b'?')) + 16);
            rest = &rest[ch.len_utf8()..];
        }
        Ok(tokens)
    }

    fn code_piece(&self, token: i32) -> Option<String> {
        match token {
            WORDS..400 => Some(format!("w{}", token - WORDS)),
            SPACED_WORDS..500 => Some(format!("w{}", token - SPACED_WORDS)),
            SPACED_CHARS..212 => Some(ALNUM[(token - SPACED_CHARS) as usize..][..1].to_string()),
            16..144 => {
                let ch = char::from(u8::try_from(token - 16).ok()?);
                ch.is_ascii_alphanumeric().then(|| ch.to_string())
            }
            _ => None,
        }
    }
}

pub(crate) struct FakeModel {
    pub info: ModelInfo,
    pub chat: ChatFormat,
    pub log: Rc<RefCell<Log>>,
    /// Token favored at each canvas row of full logits, by row index.
    pub favored: Vec<i32>,
    /// Causal continuation after the last `<think>`: the prediction for thought position `k`
    /// is `causal[k]` whatever the earlier tokens are, or `<end>` past the script.
    pub causal: Vec<i32>,
    pub scheme: DiffusionScheme,
    pub tokenizer: FakeTokenizer,
    profile: PrefillProfile,
    rows: usize,
    prompt: Vec<i32>,
}

impl FakeModel {
    pub fn new() -> Self {
        Self {
            info: ModelInfo {
                architecture: "fake",
                display_name: "Fake model".into(),
                n_vocab: VOCAB,
                n_ctx: 256,
                batch_size: 64,
                max_canvas: 64,
            },
            chat: ChatFormat {
                bos: true,
                user_open: "<user>".into(),
                model_open: "<end><model>".into(),
                thought_open: "<think>".into(),
                thought_close: "</think>".into(),
                empty_thought: "<nothought>".into(),
                thought_stops: vec!["</think>".into(), "<end>".into()],
                space_joins_answers: false,
            },
            log: Rc::default(),
            favored: Vec::new(),
            causal: Vec::new(),
            scheme: DiffusionScheme::UniformSelfConditioned { mask: MASK },
            tokenizer: FakeTokenizer { space_joins: false },
            profile: PrefillProfile::default(),
            rows: 0,
            prompt: Vec::new(),
        }
    }
}

impl DiffusionModel for FakeModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenizer(&self) -> &dyn TextTokenizer {
        &self.tokenizer
    }

    fn chat(&self) -> &ChatFormat {
        &self.chat
    }

    fn scheme(&self) -> DiffusionScheme {
        self.scheme
    }

    fn encode_images(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
        if images.is_empty() {
            Ok(Vec::new())
        } else {
            Err(Error::InvalidInput("no images".into()))
        }
    }

    fn prefill(&mut self, parts: &[PromptPart], suffix: &[i32]) -> Result<usize> {
        let mut prompt = Vec::new();
        for part in parts {
            match part {
                PromptPart::Text(tokens) => prompt.extend(tokens),
                PromptPart::Image { .. } => unreachable!(),
            }
        }
        prompt.extend(suffix);
        let reused = prompt
            .iter()
            .zip(&self.prompt)
            .take_while(|(a, b)| a == b)
            .count();
        self.profile.calls += 1;
        self.profile.batches += 1;
        self.profile.reused_tokens += reused;
        self.profile.processed_tokens += prompt.len() - reused;
        self.log.borrow_mut().prefills.push(prompt.clone());
        self.prompt = prompt;
        Ok(self.prompt.len())
    }

    fn forward_canvas(
        &mut self,
        tokens: &[i32],
        prompt_length: usize,
        conditioning: Conditioning<'_>,
        _logits: Logits,
    ) -> Result<()> {
        assert_eq!(prompt_length, self.prompt.len());
        if matches!(self.scheme, DiffusionScheme::Masked { .. }) {
            assert!(matches!(conditioning, Conditioning::None));
        }
        let mut log = self.log.borrow_mut();
        log.canvases.push(tokens.to_vec());
        log.conditioned
            .push(matches!(conditioning, Conditioning::Previous { .. }));
        self.rows = tokens.len();
        Ok(())
    }

    /// Candidate `i` gets logit `-i`, shifted by the row so rows differ.
    fn candidate_logits(&mut self, row: usize, candidates: &[i32]) -> Result<Vec<f64>> {
        assert!(row < self.rows);
        Ok((0..candidates.len())
            .map(|i| row as f64 * 0.01 - i as f64)
            .collect())
    }

    fn full_logits(&mut self) -> Result<Vec<f32>> {
        let vocab = VOCAB as usize;
        let mut logits = vec![0.0; self.rows * vocab];
        for row in 0..self.rows {
            let token = self.favored.get(row).copied().unwrap_or(20 + row as i32);
            logits[row * vocab + token as usize] = 100.0;
        }
        Ok(logits)
    }

    fn prefill_predict(
        &mut self,
        parts: &[PromptPart],
        suffix: &[i32],
        rows: usize,
    ) -> Result<(usize, Vec<i32>)> {
        let length = self.prefill(parts, suffix)?;
        assert!((1..=length).contains(&rows));
        self.log.borrow_mut().predicted.push(rows);
        let start = self
            .prompt
            .iter()
            .rposition(|&t| t == 12)
            .map_or(length, |i| i + 1);
        let predictions = (length - rows..length)
            .map(|position| {
                let k = (position + 1).saturating_sub(start);
                self.causal.get(k).copied().unwrap_or(14)
            })
            .collect();
        Ok((length, predictions))
    }

    fn profile(&mut self) -> &mut PrefillProfile {
        &mut self.profile
    }
}
