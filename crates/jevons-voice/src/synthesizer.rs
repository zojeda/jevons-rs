//! Speaking a text over any [`VoiceModel`]: the text is cleaned, cut into sentences, each
//! sentence gets its language and its numbers in words, and the model speaks one at a time. The
//! first audio then leaves after the first sentence, and a model stays in the length where it
//! does not skip or invent words.
use crate::sentences::MAX_SENTENCE_CHARS;
use crate::{clean, language, numbers, sentences};
use jevons_core::{Error, Result, Utterance, VoiceInfo, VoiceModel};

/// A text to speak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeakRequest<'a> {
    pub text: &'a str,
    /// One of the model's voices. `None` speaks in the default voice.
    pub voice: Option<&'a str>,
    /// One of the model's languages, for the whole text. `None` guesses each sentence's.
    pub language: Option<&'a str>,
}

impl<'a> SpeakRequest<'a> {
    /// `text` in the default voice, each sentence in the language its words point to.
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            voice: None,
            language: None,
        }
    }
}

/// What a call to [`Synthesizer::speak`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Spoken {
    /// Sentences the model was asked to speak, the one a stop cut short included.
    pub sentences: usize,
    /// Samples handed to the sink, at the model's rate.
    pub samples: usize,
    /// The sink returned false, and the rest of the text stayed unspoken.
    pub stopped: bool,
}

/// One stretch of text ready for the model.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Sentence {
    text: String,
    language: Option<&'static str>,
}

pub struct Synthesizer {
    model: Box<dyn VoiceModel>,
    default_voice: String,
}

impl Synthesizer {
    /// Wraps `model`. `default_voice` is the voice of a request that names none: one of the
    /// model's, or `None` for the first the model lists.
    pub fn new(model: Box<dyn VoiceModel>, default_voice: Option<&str>) -> Result<Self> {
        let info = model.info();
        let default_voice = match default_voice {
            Some(voice) => known_voice(info, voice)?.to_string(),
            None => info.voices.first().cloned().ok_or_else(|| {
                Error::InvalidInput(format!("{} has no voices", info.display_name))
            })?,
        };
        Ok(Self {
            model,
            default_voice,
        })
    }

    /// Detects and loads the voice model named by `config` on the calling thread.
    #[cfg(feature = "models")]
    pub fn load(config: &jevons_core::VoiceConfig, default_voice: Option<&str>) -> Result<Self> {
        Self::new(jevons_models::load_voice(config)?, default_voice)
    }

    pub fn info(&self) -> &VoiceInfo {
        self.model.info()
    }

    /// The voice of a request that names none.
    pub fn default_voice(&self) -> &str {
        &self.default_voice
    }

    /// Speaks `request.text`, one sentence per call to the model. `audio` receives mono samples
    /// at the model's rate as the model produces them, and returns false to stop: nothing more
    /// is spoken, and [`Spoken::stopped`] says so. A voice or a language the model does not
    /// have, and a text with nothing to say, are [`Error::InvalidInput`] before any audio.
    pub fn speak(
        &mut self,
        request: &SpeakRequest,
        audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<Spoken> {
        let info = self.model.info();
        let voice = match request.voice {
            Some(voice) => known_voice(info, voice)?,
            None => &self.default_voice,
        };
        let language = request
            .language
            .map(|language| known_language(info, language))
            .transpose()?;
        let sentences = prepare(request.text, language, info.languages);
        if sentences.is_empty() {
            return Err(Error::InvalidInput("The text has nothing to say".into()));
        }
        let mut spoken = Spoken::default();
        for sentence in &sentences {
            spoken.sentences += 1;
            let utterance = Utterance {
                text: &sentence.text,
                voice,
                language: sentence.language,
            };
            self.model.speak(&utterance, &mut |samples| {
                // A model that goes on after a stop reaches the sink no more.
                if spoken.stopped {
                    return false;
                }
                spoken.samples += samples.len();
                spoken.stopped = !audio(samples);
                !spoken.stopped
            })?;
            if spoken.stopped {
                break;
            }
        }
        Ok(spoken)
    }
}

/// `voice`, when the model lists it.
fn known_voice<'a>(info: &VoiceInfo, voice: &'a str) -> Result<&'a str> {
    if info.voices.iter().any(|known| known == voice) {
        return Ok(voice);
    }
    Err(Error::InvalidInput(format!(
        "Unknown voice {voice:?}: {} has {}",
        info.display_name,
        info.voices.join(", ")
    )))
}

/// `language` as the model lists it.
fn known_language(info: &VoiceInfo, language: &str) -> Result<&'static str> {
    info.languages
        .iter()
        .copied()
        .find(|known| *known == language)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unsupported language {language:?}: {} speaks {}",
                info.display_name,
                info.languages.join(", ")
            ))
        })
}

/// The utterances of `text`, in order: cleaned, cut at sentence ends, each with its language
/// and its numbers in that language's words, and cut again where one is longer than
/// [`MAX_SENTENCE_CHARS`]. `forced` is the request's language. Without it a sentence takes the
/// language its own words point to among `languages`, or else the sentence before's.
fn prepare(
    text: &str,
    forced: Option<&'static str>,
    languages: &'static [&'static str],
) -> Vec<Sentence> {
    let text = clean::clean(text);
    // Before a sentence says otherwise: the language the whole text points to, then the first
    // one the model lists that the service tells apart, then the model's first.
    let known = || {
        languages
            .iter()
            .copied()
            .find(|l| language::KNOWN.contains(l))
    };
    let mut current = forced
        .or_else(|| language::guess(&text, languages))
        .or_else(known)
        .or_else(|| languages.first().copied());
    let mut prepared = Vec::new();
    for sentence in sentences::split(&text) {
        if forced.is_none() {
            current = language::guess(sentence, languages).or(current);
        }
        let spelled = numbers::expand(sentence, current);
        for piece in sentences::fit(&spelled, MAX_SENTENCE_CHARS) {
            // Marks alone are nothing to say.
            if piece.contains(char::is_alphanumeric) {
                prepared.push(Sentence {
                    text: piece.to_string(),
                    language: current,
                });
            }
        }
    }
    prepared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeVoice, Log, Spoke, fake_synthesizer, samples_for, spoken};

    /// What the model was asked to speak: text, voice, language.
    fn asked(log: &Log) -> Vec<(String, String, String)> {
        spoken(log)
            .into_iter()
            .map(|spoke| (spoke.text, spoke.voice, spoke.language.unwrap_or_default()))
            .collect()
    }

    fn said(log: &Log) -> Vec<String> {
        spoken(log).into_iter().map(|spoke| spoke.text).collect()
    }

    fn row(text: &str, voice: &str, language: &str) -> (String, String, String) {
        (text.into(), voice.into(), language.into())
    }

    /// A synthesizer over a fake that speaks `languages`.
    fn speaking(languages: &'static [&'static str]) -> (Synthesizer, Log) {
        let mut model = FakeVoice::new();
        model.info.languages = languages;
        let log = model.log.clone();
        (Synthesizer::new(Box::new(model), None).unwrap(), log)
    }

    fn invalid(result: Result<Spoken>) -> String {
        match result {
            Err(Error::InvalidInput(message)) => message,
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    #[test]
    fn each_sentence_is_one_utterance_in_the_default_voice() {
        let (mut synthesizer, log) = fake_synthesizer();
        let text = "Hola. ¿Cómo estás? Todo bien por acá.";
        let spoken = synthesizer
            .speak(&SpeakRequest::new(text), &mut |_| true)
            .unwrap();
        assert_eq!(
            asked(&log),
            vec![
                row("Hola.", "lector", "es"),
                row("¿Cómo estás?", "lector", "es"),
                row("Todo bien por acá.", "lector", "es"),
            ]
        );
        let samples: usize = said(&log).iter().map(|s| samples_for(s, 24000)).sum();
        assert_eq!(
            spoken,
            Spoken {
                sentences: 3,
                samples,
                stopped: false
            }
        );
    }

    #[test]
    fn the_first_audio_leaves_before_the_second_sentence_is_asked() {
        let (mut synthesizer, log) = fake_synthesizer();
        let mut asked_at_each_chunk = Vec::new();
        synthesizer
            .speak(&SpeakRequest::new("Uno. Dos."), &mut |_| {
                asked_at_each_chunk.push(spoken(&log).len());
                true
            })
            .unwrap();
        // "Uno." and "Dos." are two chunks each.
        assert_eq!(asked_at_each_chunk, vec![1, 1, 2, 2]);
    }

    #[test]
    fn the_default_voice_is_the_models_first_unless_one_is_given() {
        let (synthesizer, _) = fake_synthesizer();
        assert_eq!(synthesizer.default_voice(), "lector");
        assert_eq!(synthesizer.info().sample_rate, 24000);
        let model = FakeVoice::new();
        let log = model.log.clone();
        let mut synthesizer = Synthesizer::new(Box::new(model), Some("reader")).unwrap();
        assert_eq!(synthesizer.default_voice(), "reader");
        synthesizer
            .speak(&SpeakRequest::new("Hello there."), &mut |_| true)
            .unwrap();
        assert_eq!(asked(&log), vec![row("Hello there.", "reader", "en")]);
    }

    #[test]
    fn a_default_voice_the_model_lacks_is_invalid_input() {
        let result = Synthesizer::new(Box::new(FakeVoice::new()), Some("nadie"));
        assert!(
            matches!(&result, Err(Error::InvalidInput(message)) if message.contains("nadie")),
            "{:?}",
            result.err()
        );
        let mut silent = FakeVoice::new();
        silent.info.voices.clear();
        let result = Synthesizer::new(Box::new(silent), None);
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn a_requested_voice_must_be_one_the_model_has() {
        let (mut synthesizer, log) = fake_synthesizer();
        let request = SpeakRequest {
            voice: Some("reader"),
            ..SpeakRequest::new("Hola a todos.")
        };
        synthesizer.speak(&request, &mut |_| true).unwrap();
        assert_eq!(asked(&log), vec![row("Hola a todos.", "reader", "es")]);
        let request = SpeakRequest {
            voice: Some("nadie"),
            ..SpeakRequest::new("Hola a todos.")
        };
        let mut heard = 0;
        let message = invalid(synthesizer.speak(&request, &mut |samples| {
            heard += samples.len();
            true
        }));
        assert!(
            message.contains("nadie") && message.contains("lector, reader"),
            "{message}"
        );
        // Nothing reached the model or the sink.
        assert_eq!((heard, spoken(&log).len()), (0, 1));
    }

    #[test]
    fn a_requested_language_holds_for_every_sentence_and_must_be_the_models() {
        let (mut synthesizer, log) = fake_synthesizer();
        let request = SpeakRequest {
            language: Some("en"),
            ..SpeakRequest::new("El total es 21. ¿Está bien?")
        };
        synthesizer.speak(&request, &mut |_| true).unwrap();
        assert_eq!(
            asked(&log),
            vec![
                row("El total es twenty-one.", "lector", "en"),
                row("¿Está bien?", "lector", "en"),
            ]
        );
        let request = SpeakRequest {
            language: Some("fr"),
            ..SpeakRequest::new("Bonjour.")
        };
        let message = invalid(synthesizer.speak(&request, &mut |_| true));
        assert!(
            message.contains("fr") && message.contains("es, en"),
            "{message}"
        );
        assert_eq!(spoken(&log).len(), 2);
    }

    #[test]
    fn each_sentence_takes_the_language_of_its_words_and_its_number_rules() {
        let (mut synthesizer, log) = fake_synthesizer();
        let text = "El total es 1.500 pesos. The total is 1,500 dollars. Son 3,5 kilos.";
        synthesizer
            .speak(&SpeakRequest::new(text), &mut |_| true)
            .unwrap();
        assert_eq!(
            asked(&log),
            vec![
                row("El total es mil quinientos pesos.", "lector", "es"),
                row(
                    "The total is one thousand five hundred dollars.",
                    "lector",
                    "en"
                ),
                row("Son tres coma cinco kilos.", "lector", "es"),
            ]
        );
    }

    #[test]
    fn a_sentence_without_evidence_takes_the_language_of_the_one_before() {
        let (mut synthesizer, log) = fake_synthesizer();
        let text = "This is the first one. 42. Perfecto. Y este es el último. 42.";
        synthesizer
            .speak(&SpeakRequest::new(text), &mut |_| true)
            .unwrap();
        assert_eq!(
            asked(&log),
            vec![
                row("This is the first one.", "lector", "en"),
                row("Forty-two.", "lector", "en"),
                row("Perfecto.", "lector", "en"),
                row("Y este es el último.", "lector", "es"),
                row("Cuarenta y dos.", "lector", "es"),
            ]
        );
    }

    #[test]
    fn a_first_sentence_without_evidence_takes_the_texts_language_then_a_known_one() {
        // The whole text points to English.
        let (mut synthesizer, log) = fake_synthesizer();
        synthesizer
            .speak(&SpeakRequest::new("OK. This is what I found."), &mut |_| {
                true
            })
            .unwrap();
        assert_eq!(
            asked(&log),
            vec![
                row("OK.", "lector", "en"),
                row("This is what I found.", "lector", "en"),
            ]
        );
        // Nothing points anywhere: the model's first language the service tells apart, which
        // is not always the model's first.
        let (mut synthesizer, log) = speaking(&["zh", "en", "de", "es"]);
        synthesizer
            .speak(&SpeakRequest::new("OK. 42."), &mut |_| true)
            .unwrap();
        assert_eq!(
            asked(&log),
            vec![
                row("OK.", "lector", "en"),
                row("Forty-two.", "lector", "en")
            ]
        );
        let (mut synthesizer, log) = fake_synthesizer();
        synthesizer
            .speak(&SpeakRequest::new("42."), &mut |_| true)
            .unwrap();
        assert_eq!(asked(&log), vec![row("Cuarenta y dos.", "lector", "es")]);
    }

    #[test]
    fn a_model_without_spanish_or_english_gets_its_first_language_and_the_digits() {
        let (mut synthesizer, log) = speaking(&["fr", "de"]);
        synthesizer
            .speak(&SpeakRequest::new("J'ai **3** chats."), &mut |_| true)
            .unwrap();
        assert_eq!(asked(&log), vec![row("J'ai 3 chats.", "lector", "fr")]);
        // A model that lists no language is left to choose.
        let (mut synthesizer, log) = speaking(&[]);
        synthesizer
            .speak(&SpeakRequest::new("Tengo 3 gatos."), &mut |_| true)
            .unwrap();
        assert_eq!(
            spoken(&log),
            vec![Spoke {
                text: "Tengo 3 gatos.".into(),
                voice: "lector".into(),
                language: None,
            }]
        );
    }

    #[test]
    fn markdown_is_cleaned_before_the_model_reads() {
        let (mut synthesizer, log) = fake_synthesizer();
        let text = "## Resumen\n\n- **Total:** $1.500 😀\n- Más en [la guía](https://example.com/x).\n\n```\nlet x = 1;\n```";
        synthesizer
            .speak(&SpeakRequest::new(text), &mut |_| true)
            .unwrap();
        assert_eq!(
            said(&log),
            vec!["Resumen", "Total: mil quinientos pesos", "Más en la guía."]
        );
    }

    #[test]
    fn text_with_nothing_to_say_is_invalid_input() {
        let (mut synthesizer, log) = fake_synthesizer();
        for text in [
            "",
            "   \n ",
            "...",
            "**  **",
            "😀 👍",
            "```\nlet x = 1;\n```",
            "---",
        ] {
            let message = invalid(synthesizer.speak(&SpeakRequest::new(text), &mut |_| true));
            assert!(message.contains("nothing to say"), "{text:?}: {message}");
        }
        assert!(spoken(&log).is_empty());
        // Marks alone are skipped beside a sentence.
        synthesizer
            .speak(&SpeakRequest::new("... Hola. !!!"), &mut |_| true)
            .unwrap();
        assert_eq!(said(&log), vec!["Hola."]);
    }

    #[test]
    fn a_false_sink_stops_at_once_and_is_reported() {
        let (mut synthesizer, log) = fake_synthesizer();
        let text = "Esta es la primera frase. Esta es la segunda. Y la tercera.";
        let mut chunks = 0;
        let spoken_now = synthesizer
            .speak(&SpeakRequest::new(text), &mut |_| {
                chunks += 1;
                chunks < 3
            })
            .unwrap();
        // The third chunk is still in the first sentence: 80 ms a chunk at 24 kHz.
        assert_eq!(
            spoken_now,
            Spoken {
                sentences: 1,
                samples: 3 * 1920,
                stopped: true
            }
        );
        assert_eq!(chunks, 3);
        assert_eq!(said(&log), vec!["Esta es la primera frase."]);
        // A stop on a sentence's last chunk leaves the next one unasked too.
        let (mut synthesizer, log) = fake_synthesizer();
        let mut chunks = 0;
        let spoken_now = synthesizer
            .speak(&SpeakRequest::new("Uno. Dos."), &mut |_| {
                chunks += 1;
                chunks < 2
            })
            .unwrap();
        assert_eq!((spoken_now.sentences, spoken_now.stopped), (1, true));
        assert_eq!(said(&log), vec!["Uno."]);
    }

    #[test]
    fn a_model_failure_ends_the_speech_with_its_error() {
        let mut model = FakeVoice::new();
        model.fail_on = Some("segunda".into());
        let log = model.log.clone();
        let mut synthesizer = Synthesizer::new(Box::new(model), None).unwrap();
        let mut heard = 0;
        let result = synthesizer.speak(
            &SpeakRequest::new("La primera. La segunda. La tercera."),
            &mut |samples| {
                heard += samples.len();
                true
            },
        );
        assert!(matches!(result, Err(Error::Backend(_))), "{result:?}");
        assert_eq!(heard, samples_for("La primera.", 24000));
        assert_eq!(said(&log), vec!["La primera.", "La segunda."]);
    }

    #[test]
    fn a_long_sentence_reaches_the_model_in_pieces_within_the_limit() {
        let (mut synthesizer, log) = fake_synthesizer();
        let clause = "esta es una frase que sigue y sigue sin terminar nunca";
        let text = format!("{}.", [clause; 12].join(", "));
        assert!(text.chars().count() > 2 * MAX_SENTENCE_CHARS);
        synthesizer
            .speak(&SpeakRequest::new(&text), &mut |_| true)
            .unwrap();
        let pieces = said(&log);
        assert!(pieces.len() >= 3, "{pieces:?}");
        for piece in &pieces[..pieces.len() - 1] {
            assert!(piece.chars().count() <= MAX_SENTENCE_CHARS, "{piece}");
            assert!(piece.ends_with(','), "{piece}");
        }
        assert_eq!(pieces.join(" "), text);
        assert!(asked(&log).iter().all(|(_, _, language)| language == "es"));
    }

    #[test]
    fn the_limit_applies_to_a_sentence_with_its_numbers_in_words() {
        // Forty numbers in 159 characters of digits are over the limit in words.
        let text = format!("Son {}.", ["777"; 40].join(" "));
        assert!(text.chars().count() < MAX_SENTENCE_CHARS);
        let prepared = prepare(&text, None, &["es", "en"]);
        assert!(prepared.len() > 1);
        for sentence in &prepared {
            assert!(sentence.text.chars().count() <= MAX_SENTENCE_CHARS);
            assert!(!sentence.text.contains(|c: char| c.is_ascii_digit()));
        }
    }
}
