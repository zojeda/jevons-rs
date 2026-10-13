//! A scripted voice model for the tests of this service and of the layers above it: no GPU and
//! no model files.
use crate::Synthesizer;
use jevons_core::{Error, Result, Utterance, VoiceInfo, VoiceModel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// Seconds of audio the fake speaks for each character of an utterance.
pub const SECONDS_PER_CHAR: f64 = 0.04;
/// Seconds of audio in each chunk the fake hands over. The last one may be shorter.
pub const CHUNK_SECONDS: f64 = 0.08;
/// The peak of the fake's tone.
pub const LEVEL: f32 = 0.1;
/// The pitch of the fake's tone, in Hz.
pub const TONE_HZ: f64 = 220.0;

/// One utterance the fake was asked to speak.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spoke {
    pub text: String,
    pub voice: String,
    pub language: Option<String>,
}

/// Every utterance the fake was asked to speak, in call order. A test keeps a clone while the
/// model lives on its worker thread.
pub type Log = Arc<Mutex<Vec<Spoke>>>;

/// The utterances in `log` so far.
pub fn spoken(log: &Log) -> Vec<Spoke> {
    log.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

pub struct FakeVoice {
    pub info: VoiceInfo,
    pub log: Log,
    /// An utterance whose text contains this fails with [`Error::Backend`] before any audio.
    pub fail_on: Option<String>,
    /// How long the fake sleeps before each chunk, so a test has time to stop it.
    pub chunk_delay: Duration,
}

impl FakeVoice {
    /// A 24 kHz model that speaks Spanish and English in the voices `lector` and `reader`,
    /// never fails and never sleeps.
    pub fn new() -> Self {
        Self {
            info: VoiceInfo {
                architecture: "fake",
                display_name: "Fake voice".into(),
                sample_rate: 24000,
                languages: &["es", "en"],
                voices: vec!["lector".into(), "reader".into()],
            },
            log: Log::default(),
            fail_on: None,
            chunk_delay: Duration::ZERO,
        }
    }
}

impl Default for FakeVoice {
    fn default() -> Self {
        Self::new()
    }
}

/// How many samples the fake speaks for `text` at `sample_rate`.
pub fn samples_for(text: &str, sample_rate: u32) -> usize {
    let seconds = text.chars().count() as f64 * SECONDS_PER_CHAR;
    (seconds * f64::from(sample_rate)).round() as usize
}

impl VoiceModel for FakeVoice {
    fn info(&self) -> &VoiceInfo {
        &self.info
    }

    fn speak(
        &mut self,
        utterance: &Utterance,
        audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Spoke {
                text: utterance.text.into(),
                voice: utterance.voice.into(),
                language: utterance.language.map(Into::into),
            });
        if !self
            .info
            .voices
            .iter()
            .any(|voice| voice == utterance.voice)
        {
            return Err(Error::InvalidInput(format!(
                "Unknown voice {:?}",
                utterance.voice
            )));
        }
        if let Some(language) = utterance.language
            && !self.info.languages.contains(&language)
        {
            return Err(Error::InvalidInput(format!(
                "Unsupported language {language:?}"
            )));
        }
        if let Some(text) = &self.fail_on
            && utterance.text.contains(text.as_str())
        {
            return Err(Error::Backend("the fake voice was told to fail".into()));
        }
        let rate = f64::from(self.info.sample_rate);
        let total = samples_for(utterance.text, self.info.sample_rate);
        let chunk = ((CHUNK_SECONDS * rate).round() as usize).max(1);
        let mut start = 0;
        while start < total {
            if !self.chunk_delay.is_zero() {
                std::thread::sleep(self.chunk_delay);
            }
            let end = (start + chunk).min(total);
            // One tone across the utterance: each chunk goes on where the last one stopped.
            let samples: Vec<f32> = (start..end)
                .map(|n| {
                    let phase = 2.0 * std::f64::consts::PI * TONE_HZ * n as f64 / rate;
                    LEVEL * phase.sin() as f32
                })
                .collect();
            if !audio(&samples) {
                break;
            }
            start = end;
        }
        Ok(())
    }
}

/// A synthesizer over [`FakeVoice::new`] with the model's first voice as its default, and the
/// log of what the model is asked to speak.
pub fn fake_synthesizer() -> (Synthesizer, Log) {
    let model = FakeVoice::new();
    let log = model.log.clone();
    let synthesizer =
        Synthesizer::new(Box::new(model), None).expect("the fake voice model has voices");
    (synthesizer, log)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterance(text: &str) -> Utterance<'_> {
        Utterance {
            text,
            voice: "lector",
            language: Some("es"),
        }
    }

    #[test]
    fn the_fake_speaks_a_quiet_tone_of_40_ms_a_character_in_chunks_of_80_ms() {
        let mut model = FakeVoice::new();
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        model
            .speak(&utterance("Hola."), &mut |samples| {
                chunks.push(samples.to_vec());
                true
            })
            .unwrap();
        // Five characters are 200 ms: two chunks of 80 ms and one of 40 ms, at 24 kHz.
        let lengths: Vec<usize> = chunks.iter().map(Vec::len).collect();
        assert_eq!(lengths, vec![1920, 1920, 960]);
        assert_eq!(samples_for("Hola.", 24000), 4800);
        let samples = chunks.concat();
        let peak = samples.iter().fold(0f32, |peak, s| peak.max(s.abs()));
        assert!(peak > 0.09 && peak <= LEVEL, "{peak}");
        // The same text gives the same audio, and the tone does not restart at a chunk.
        let mut again = Vec::new();
        model
            .speak(&utterance("Hola."), &mut |samples| {
                again.extend_from_slice(samples);
                true
            })
            .unwrap();
        assert_eq!(again, samples);
        let step = (samples[1920] - samples[1919]).abs();
        assert!(step < 0.01, "{step}");
        // Another rate keeps the durations.
        model.info.sample_rate = 16000;
        let mut lengths = Vec::new();
        model
            .speak(&utterance("Hola."), &mut |samples| {
                lengths.push(samples.len());
                true
            })
            .unwrap();
        assert_eq!(lengths, vec![1280, 1280, 640]);
    }

    #[test]
    fn the_fake_logs_what_it_is_asked_and_rejects_what_it_does_not_have() {
        let mut model = FakeVoice::new();
        let log = model.log.clone();
        model.speak(&utterance("Uno."), &mut |_| true).unwrap();
        let english = Utterance {
            text: "Two.",
            voice: "reader",
            language: None,
        };
        model.speak(&english, &mut |_| true).unwrap();
        assert_eq!(
            spoken(&log),
            vec![
                Spoke {
                    text: "Uno.".into(),
                    voice: "lector".into(),
                    language: Some("es".into()),
                },
                Spoke {
                    text: "Two.".into(),
                    voice: "reader".into(),
                    language: None,
                },
            ]
        );
        let mut heard = 0;
        let stranger = Utterance {
            voice: "nadie",
            ..utterance("Tres.")
        };
        let foreign = Utterance {
            language: Some("fr"),
            ..utterance("Quatre.")
        };
        for unknown in [stranger, foreign] {
            let result = model.speak(&unknown, &mut |samples| {
                heard += samples.len();
                true
            });
            assert!(matches!(result, Err(Error::InvalidInput(_))), "{result:?}");
        }
        assert_eq!(heard, 0);
        assert_eq!(spoken(&log).len(), 4);
    }

    #[test]
    fn the_fake_fails_on_its_scripted_text_and_stops_when_told() {
        let mut model = FakeVoice::new();
        model.fail_on = Some("falla".into());
        let mut heard = 0;
        let result = model.speak(&utterance("Esto falla siempre."), &mut |samples| {
            heard += samples.len();
            true
        });
        assert!(matches!(result, Err(Error::Backend(_))), "{result:?}");
        assert_eq!(heard, 0);
        // A sink that returns false gets no further chunk.
        let mut chunks = 0;
        model
            .speak(&utterance("Esto no, y es largo."), &mut |_| {
                chunks += 1;
                chunks < 2
            })
            .unwrap();
        assert_eq!(chunks, 2);
    }

    #[test]
    fn the_fake_waits_before_each_chunk() {
        let mut model = FakeVoice::new();
        model.chunk_delay = Duration::from_millis(20);
        let started = std::time::Instant::now();
        let mut chunks = 0;
        model
            .speak(&utterance("Hola."), &mut |_| {
                chunks += 1;
                true
            })
            .unwrap();
        assert_eq!(chunks, 3);
        assert!(started.elapsed() >= Duration::from_millis(60));
    }
}
