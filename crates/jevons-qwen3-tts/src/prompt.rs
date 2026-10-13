//! The talker's prompt, as `qwen-tts` lays it out for a voice given by its speaker vector.
//!
//! Each position sums a text side and a codec side. The text is
//! `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`: the role opens the
//! prompt, the codec prefix (the language, the speaker, the start of audio) rides on text
//! padding, and the text's first token rides on the start-of-audio code. The rest of the
//! text enters one token per generated frame, then the end-of-text mark, then padding.
use crate::config::Config;

/// The text side of a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Text {
    /// A token of the text tokenizer.
    Token(i32),
    Pad,
    Bos,
    Eos,
}

/// The codec side of a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    None,
    /// An id of the talker's codec vocabulary.
    Id(i32),
    /// The voice's speaker vector.
    Speaker,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Prompt {
    /// The positions the talker reads before it writes its first code.
    pub prefill: Vec<(Text, Codec)>,
    /// The text side of each generated frame's position, in order. After it comes padding.
    pub trailing: Vec<Text>,
}

/// The prompt for `text` (its token ids, at least one), spoken in the language `language`
/// names (a codec id) or in whatever language the talker reads into the text.
pub(crate) fn prompt(config: &Config, newline: i32, text: &[i32], language: Option<i32>) -> Prompt {
    let t = &config.talker_config;
    let (first, rest) = text.split_first().expect("a text of at least one token");
    let mut prefill: Vec<(Text, Codec)> =
        [config.im_start_token_id, config.assistant_token_id, newline]
            .into_iter()
            .map(|id| (Text::Token(id), Codec::None))
            .collect();
    let thought = match language {
        Some(language) => vec![
            t.codec_think_id,
            t.codec_think_bos_id,
            language,
            t.codec_think_eos_id,
        ],
        None => vec![
            t.codec_nothink_id,
            t.codec_think_bos_id,
            t.codec_think_eos_id,
        ],
    };
    prefill.extend(thought.into_iter().map(|id| (Text::Pad, Codec::Id(id))));
    prefill.push((Text::Pad, Codec::Speaker));
    prefill.push((Text::Bos, Codec::Id(t.codec_pad_id)));
    prefill.push((Text::Token(*first), Codec::Id(t.codec_bos_id)));
    let mut trailing: Vec<Text> = rest.iter().map(|id| Text::Token(*id)).collect();
    trailing.push(Text::Eos);
    Prompt { prefill, trailing }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::CONFIG;

    #[test]
    fn a_prompt_names_the_language_then_the_speaker_then_starts_the_audio() {
        let config: Config = serde_json::from_str(CONFIG).unwrap();
        let got = prompt(&config, 198, &[39, 2253, 13], Some(2054));
        let token = |id| Text::Token(id);
        assert_eq!(
            got.prefill,
            vec![
                (token(151644), Codec::None),
                (token(77091), Codec::None),
                (token(198), Codec::None),
                (Text::Pad, Codec::Id(2154)),
                (Text::Pad, Codec::Id(2156)),
                (Text::Pad, Codec::Id(2054)),
                (Text::Pad, Codec::Id(2157)),
                (Text::Pad, Codec::Speaker),
                (Text::Bos, Codec::Id(2148)),
                (token(39), Codec::Id(2149)),
            ]
        );
        assert_eq!(got.trailing, vec![token(2253), token(13), Text::Eos]);
    }

    #[test]
    fn a_prompt_without_a_language_skips_the_thought() {
        let config: Config = serde_json::from_str(CONFIG).unwrap();
        let got = prompt(&config, 198, &[39], None);
        assert_eq!(got.prefill.len(), 9);
        assert_eq!(got.prefill[3], (Text::Pad, Codec::Id(2155)));
        assert_eq!(got.prefill[5], (Text::Pad, Codec::Id(2157)));
        assert_eq!(got.trailing, vec![Text::Eos]);
    }
}
