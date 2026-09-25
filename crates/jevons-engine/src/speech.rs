//! Transcription of whole recordings over any [`SpeechModel`]: overlapping windows for audio
//! longer than one model pass, then words and segments with timestamps.
//!
//! Windows overlap by [`CONTEXT_SECONDS`] on each side. Each window keeps the words that start
//! in its central span, so every word is decoded with context on both sides and appears once.
use jevons_core::{Result, Segment, SpeechInfo, SpeechModel, SpeechToken, Transcript, Word};

/// Audio each window sees past the span it keeps, on each side.
pub const CONTEXT_SECONDS: f64 = 5.0;
/// A pause this long ends a segment.
const SEGMENT_PAUSE_SECONDS: f64 = 0.8;
/// Segments are split before they grow longer than this.
const SEGMENT_MAX_SECONDS: f64 = 30.0;

/// Groups tokens into words at `▁` word starts. Leading tokens without one form a word.
pub fn words(tokens: &[SpeechToken]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for token in tokens {
        let starts_word = token.piece.starts_with('▁');
        let text = token.piece.replace('▁', " ");
        match words.last_mut() {
            Some(word) if !starts_word => {
                word.text.push_str(&text);
                word.end = token.end;
                word.tokens.push(token.clone());
            }
            _ => words.push(Word {
                text,
                start: token.start,
                end: token.end,
                tokens: vec![token.clone()],
            }),
        }
    }
    for word in &mut words {
        word.text = word.text.trim().to_string();
    }
    words.retain(|w| !w.text.is_empty());
    words
}

/// Titles and abbreviations whose period does not end a sentence (English, Spanish, French,
/// German, Italian, Portuguese).
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "sra", "srta", "dra", "mme", "mlle", "hr",
    "sig", "dott", "vs", "nº", "e.g", "i.e", "z.b",
];

fn ends_sentence(word: &str) -> bool {
    if word.ends_with(['?', '!', '…', '。']) {
        return true;
    }
    let Some(stem) = word.strip_suffix('.') else {
        return false;
    };
    !ABBREVIATIONS.contains(&stem.to_lowercase().as_str())
}

/// Where to cut `words` into segments: after sentence punctuation, before a pause, or before
/// the segment would pass the length limit. Returns the end index of each segment.
fn segment_ends(words: &[Word]) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut start = 0;
    for i in 0..words.len() {
        let next = words.get(i + 1);
        let sentence = ends_sentence(&words[i].text);
        let pause = next.is_some_and(|n| n.start - words[i].end >= SEGMENT_PAUSE_SECONDS);
        let long = next.is_some_and(|n| n.end - words[start].start > SEGMENT_MAX_SECONDS);
        if next.is_none() || sentence || pause || long {
            ends.push(i + 1);
            start = i + 1;
        }
    }
    ends
}

pub struct Transcriber {
    model: Box<dyn SpeechModel>,
}

impl Transcriber {
    pub fn new(model: Box<dyn SpeechModel>) -> Self {
        Self { model }
    }

    /// Detects and loads the speech model named by `config` on the calling thread.
    #[cfg(feature = "models")]
    pub fn load(config: &jevons_core::SpeechConfig) -> Result<Self> {
        Ok(Self::new(jevons_models::load_speech(config)?))
    }

    pub fn info(&self) -> &SpeechInfo {
        self.model.info()
    }

    /// Tokens of one pass over at most [`SpeechInfo::max_window_seconds`] of audio.
    pub fn pass(&mut self, samples: &[f32]) -> Result<Vec<SpeechToken>> {
        self.model.transcribe(samples)
    }

    /// The text of `words`, with the model's detokenization.
    pub fn text(&self, words: &[Word]) -> Result<String> {
        let ids: Vec<u32> = words
            .iter()
            .flat_map(|w| w.tokens.iter().map(|t| t.id))
            .collect();
        Ok(self.model.detokenize(&ids)?.trim().to_string())
    }

    fn segment(&self, id: usize, words: &[Word]) -> Result<Segment> {
        let tokens: Vec<&SpeechToken> = words.iter().flat_map(|w| &w.tokens).collect();
        let logprob = tokens.iter().map(|t| t.logprob).sum::<f32>() / tokens.len().max(1) as f32;
        Ok(Segment {
            id,
            start: words.first().map_or(0.0, |w| w.start),
            end: words.last().map_or(0.0, |w| w.end),
            text: self.text(words)?,
            tokens: tokens.iter().map(|t| t.id).collect(),
            avg_logprob: logprob,
        })
    }

    /// Decodes the next window of `job` and returns the segments it closed, each with its
    /// words. Segments are closed once no later window can change them.
    pub fn step(&mut self, job: &mut Transcription) -> Result<Vec<(Segment, Vec<Word>)>> {
        assert!(!job.done, "the transcription is finished");
        let info = self.model.info();
        let rate = f64::from(info.sample_rate);
        let duration = job.samples.len() as f64 / rate;
        let window = info.max_window_seconds;
        let stride = window - 2.0 * CONTEXT_SECONDS;
        assert!(
            stride > 0.0,
            "the model window must exceed twice the context"
        );

        let begin = job.begin;
        let last = begin + window >= duration;
        let from = (begin * rate) as usize;
        let to = if last {
            job.samples.len()
        } else {
            ((begin + window) * rate) as usize
        };
        let offset = from as f64 / rate;
        let mut tokens = self.model.transcribe(&job.samples[from..to])?;
        for token in &mut tokens {
            token.start += offset;
            token.end += offset;
        }
        let keep_from = if begin == 0.0 {
            f64::NEG_INFINITY
        } else {
            begin + CONTEXT_SECONDS
        };
        let keep_to = if last {
            f64::INFINITY
        } else {
            begin + CONTEXT_SECONDS + stride
        };
        job.kept.extend(
            words(&tokens)
                .into_iter()
                .filter(|w| w.start >= keep_from && w.start < keep_to),
        );

        // The last segment may continue into the next window.
        let settled = job.ends.last().copied().unwrap_or(0);
        let ends = segment_ends(&job.kept[settled..]);
        let closed = if last {
            ends.len()
        } else {
            ends.len().saturating_sub(1)
        };
        let mut out = Vec::with_capacity(closed);
        let mut start = settled;
        for &end in &ends[..closed] {
            let words = job.kept[start..settled + end].to_vec();
            let segment = self.segment(job.segments.len(), &words)?;
            start = settled + end;
            job.ends.push(start);
            job.segments.push(segment.clone());
            out.push((segment, words));
        }
        if last {
            job.done = true;
        } else {
            job.begin += stride;
        }
        Ok(out)
    }

    /// The transcript of a finished (or stopped) job.
    pub fn finish(&self, job: Transcription) -> Result<Transcript> {
        let duration = job.samples.len() as f64 / f64::from(self.model.info().sample_rate);
        Ok(Transcript {
            text: self.text(&job.kept)?,
            duration,
            words: job.kept,
            segments: job.segments,
        })
    }

    /// Transcribes a whole recording. `on_segment` receives each closed segment with its words
    /// and returns `false` to stop (the transcript then ends with that segment).
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        on_segment: &mut dyn FnMut(&Segment, &[Word]) -> bool,
    ) -> Result<Transcript> {
        let mut job = Transcription::new(samples.to_vec());
        'windows: while !job.is_done() {
            for (segment, words) in self.step(&mut job)? {
                if !on_segment(&segment, &words) {
                    job.stop_after(segment.id);
                    break 'windows;
                }
            }
        }
        self.finish(job)
    }
}

/// A windowed transcription in progress; [`Transcriber::step`] decodes one window at a time,
/// so a caller can do other work between windows.
pub struct Transcription {
    samples: Vec<f32>,
    /// Start of the next window, in seconds.
    begin: f64,
    kept: Vec<Word>,
    segments: Vec<Segment>,
    /// The end of each segment in `kept`.
    ends: Vec<usize>,
    done: bool,
}

impl Transcription {
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            begin: 0.0,
            kept: Vec::new(),
            segments: Vec::new(),
            ends: Vec::new(),
            done: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Ends the job after segment `id`, dropping later words.
    pub fn stop_after(&mut self, id: usize) {
        self.segments.truncate(id + 1);
        self.ends.truncate(id + 1);
        self.kept.truncate(self.ends.last().copied().unwrap_or(0));
        self.done = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_core::{Error, SpeechInfo};

    /// Emits one word per second of audio, `▁w{second}` at the second's start, reading the
    /// second from a marker sample so windows see absolute time.
    struct Metronome {
        info: SpeechInfo,
    }

    fn metronome(window: f64) -> Metronome {
        Metronome {
            info: SpeechInfo {
                architecture: "test",
                display_name: "test".into(),
                sample_rate: 100,
                frame_seconds: 0.01,
                max_window_seconds: window,
                languages: &["en"],
            },
        }
    }

    impl SpeechModel for Metronome {
        fn info(&self) -> &SpeechInfo {
            &self.info
        }

        fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<SpeechToken>> {
            if samples.len() as f64 > self.info.max_window_seconds * 100.0 {
                return Err(Error::InvalidInput("window too long".into()));
            }
            Ok(samples
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 100 == 0)
                .map(|(i, &second)| {
                    let at = i as f64 / 100.0;
                    let second = second as u32;
                    // Every fifth word ends a sentence.
                    let piece = if second % 5 == 4 {
                        format!("▁w{second}.")
                    } else {
                        format!("▁w{second}")
                    };
                    SpeechToken {
                        id: second,
                        piece,
                        start: at,
                        end: at + 0.5,
                        logprob: -0.1,
                    }
                })
                .collect())
        }

        fn detokenize(&self, ids: &[u32]) -> Result<String> {
            Ok(ids
                .iter()
                .map(|id| {
                    if id % 5 == 4 {
                        format!(" w{id}.")
                    } else {
                        format!(" w{id}")
                    }
                })
                .collect())
        }
    }

    /// `seconds` of audio whose samples hold their second.
    fn audio(seconds: usize) -> Vec<f32> {
        (0..seconds * 100).map(|i| (i / 100) as f32).collect()
    }

    #[test]
    fn long_audio_is_windowed_and_every_word_is_kept_once_in_order() {
        let mut transcriber = Transcriber::new(Box::new(metronome(20.0)));
        let mut streamed = Vec::new();
        let transcript = transcriber
            .transcribe(&audio(47), &mut |s, words| {
                assert_eq!(s.tokens.len(), words.len());
                streamed.push(s.clone());
                true
            })
            .unwrap();
        let seconds: Vec<u32> = transcript.tokens().map(|t| t.id).collect();
        assert_eq!(seconds, (0..47).collect::<Vec<_>>());
        for (i, word) in transcript.words.iter().enumerate() {
            assert_eq!(word.start, i as f64, "{}", word.text);
        }
        assert_eq!(transcript.duration, 47.0);
        // Five-word sentences, each streamed exactly once, in order.
        assert_eq!(streamed, transcript.segments);
        assert_eq!(transcript.segments.len(), 10);
        assert_eq!(transcript.segments[1].text, "w5 w6 w7 w8 w9.");
        assert_eq!(
            (transcript.segments[1].start, transcript.segments[1].end),
            (5.0, 9.5)
        );
        assert!(transcript.text.starts_with("w0 w1"));
    }

    #[test]
    fn short_audio_takes_one_pass() {
        let mut model = metronome(20.0);
        let tokens = model.transcribe(&audio(3)).unwrap();
        assert_eq!(tokens.len(), 3);
        let mut transcriber = Transcriber::new(Box::new(model));
        let transcript = transcriber
            .transcribe(&audio(12), &mut |_, _| true)
            .unwrap();
        assert_eq!(transcript.words.len(), 12);
        assert_eq!(transcript.segments.len(), 3);
    }

    #[test]
    fn a_false_callback_stops_after_that_segment() {
        let mut transcriber = Transcriber::new(Box::new(metronome(20.0)));
        let transcript = transcriber
            .transcribe(&audio(47), &mut |s, _| s.id < 1)
            .unwrap();
        assert_eq!(transcript.segments.len(), 2);
        assert_eq!(transcript.words.len(), 10);
    }

    #[test]
    fn words_join_continuation_pieces_and_segments_split_at_pauses() {
        let token = |piece: &str, start: f64| SpeechToken {
            id: 0,
            piece: piece.into(),
            start,
            end: start + 0.1,
            logprob: -1.0,
        };
        let words = words(&[
            token("▁Ho", 0.0),
            token("la", 0.1),
            token(",", 0.2),
            token("▁qué", 0.3),
            token("▁tal", 2.0),
        ]);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["Hola,", "qué", "tal"]);
        assert_eq!((words[0].start, words[0].end), (0.0, 0.30000000000000004));
        assert_eq!(segment_ends(&words), vec![2, 3]);
        assert!(!ends_sentence("Mr.") && !ends_sentence("Sra.") && ends_sentence("gospel."));
        assert!(ends_sentence("¿qué?") && ends_sentence("no.") && !ends_sentence("niños,"));
    }

    /// Word error rate of `got` against `want`, ignoring case and punctuation.
    fn word_error_rate(got: &str, want: &str) -> f64 {
        let words = |text: &str| -> Vec<String> {
            text.split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|w| !w.is_empty())
                .collect()
        };
        let (got, want) = (words(got), words(want));
        let mut row: Vec<usize> = (0..=got.len()).collect();
        for (i, w) in want.iter().enumerate() {
            let mut diagonal = row[0];
            row[0] = i + 1;
            for (j, g) in got.iter().enumerate() {
                let substitution = diagonal + usize::from(w != g);
                diagonal = row[j + 1];
                row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
            }
        }
        row[got.len()] as f64 / want.len() as f64
    }

    #[test]
    #[ignore = "Requires PARAKEET_MODEL, a HIP GPU and the Parakeet reference dump"]
    fn long_recordings_are_windowed_like_one_reference_pass() {
        let golden = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join(".cache/jevons/golden/parakeet-tdt-0.6b-v3");
        let samples: Vec<f32> = std::fs::read(golden.join("es_long_audio.f32"))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(golden.join("manifest.json")).unwrap()).unwrap();
        let want = manifest["clips"]["es_long"]["text"].as_str().unwrap();
        let config = jevons_core::SpeechConfig::new(std::env::var("PARAKEET_MODEL").unwrap());
        let mut transcriber = Transcriber::load(&config).unwrap();
        let window = transcriber.info().max_window_seconds;
        assert!(
            samples.len() as f64 / 16000.0 > window,
            "the clip needs two windows"
        );
        let start = std::time::Instant::now();
        let transcript = transcriber.transcribe(&samples, &mut |_, _| true).unwrap();
        let rate = word_error_rate(&transcript.text, want);
        println!(
            "{:.1} s in {:?}, {} segments, WER {rate:.3} against one reference pass",
            transcript.duration,
            start.elapsed(),
            transcript.segments.len()
        );
        assert!(rate < 0.05, "{}", transcript.text);
        assert!(
            transcript
                .words
                .windows(2)
                .all(|w| w[0].start <= w[1].start)
        );
    }

    #[test]
    fn word_error_rate_counts_edits_per_reference_word() {
        assert_eq!(word_error_rate("Hola, mundo.", "hola mundo"), 0.0);
        assert_eq!(word_error_rate("hola gran mundo", "hola mundo"), 0.5);
        assert_eq!(word_error_rate("hola", "hola mundo"), 0.5);
    }
}
