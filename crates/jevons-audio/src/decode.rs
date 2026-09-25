//! Bounded decoding of uploaded audio files to mono samples at the model rate.
//!
//! Symphonia demuxes every container and decodes most codecs; Opus packets (WebM, as browsers
//! record, and Ogg Opus) go to the `opuscule` crate, a pure-Rust decoder bit-exact with the
//! libopus reference on the RFC 8251 test vectors.
use crate::resample::Resampler;
use jevons_core::{Error, Result};
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, CODEC_TYPE_OPUS, CodecParameters, DecoderOptions};
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::{FormatOptions, Packet};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const SUPPORTED: &str =
    "Supported audio: WAV, FLAC, MP3, OGG Vorbis, Ogg Opus, WebM (Opus), M4A/MP4 (AAC)";
/// Opus always decodes here at its native rate.
const OPUS_RATE: u32 = 48_000;
/// The longest Opus packet: 120 ms at 48 kHz.
const OPUS_MAX_FRAMES: usize = 5760;

/// Opus stream parameters from its `OpusHead` (RFC 7845 §5.1), the Ogg identification header
/// and the WebM `CodecPrivate`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OpusHead {
    channels: usize,
    /// 48 kHz samples to drop from the start of the decoded stream.
    pre_skip: usize,
    /// Linear output gain.
    gain: f32,
}

impl OpusHead {
    fn parse(head: &[u8]) -> Result<Self> {
        let invalid = |detail: &str| Error::InvalidInput(format!("Invalid Opus header: {detail}"));
        if head.len() < 19 || &head[..8] != b"OpusHead" {
            return Err(invalid("missing OpusHead"));
        }
        let channels = usize::from(head[9]);
        let mapping = head[18];
        if channels == 0 || channels > 2 || (mapping != 0 && mapping != 1) {
            return Err(Error::InvalidInput(format!(
                "Opus audio with {channels} channels (mapping family {mapping}) is not \
                 supported; send mono or stereo"
            )));
        }
        if mapping == 1 && head.len() < 21 + channels {
            return Err(invalid("short channel mapping"));
        }
        if mapping == 1 && (head[19] != 1 || head[20] > 1) {
            return Err(Error::InvalidInput(
                "Multistream Opus audio is not supported; send mono or stereo".into(),
            ));
        }
        let gain_q8 = i16::from_le_bytes([head[16], head[17]]);
        Ok(Self {
            channels,
            pre_skip: usize::from(u16::from_le_bytes([head[10], head[11]])),
            gain: 10f32.powf(f32::from(gain_q8) / (20.0 * 256.0)),
        })
    }
}

/// Decodes one packet at a time to mono samples.
enum Codec {
    Symphonia {
        decoder: Box<dyn symphonia::core::codecs::Decoder>,
        buffer: Option<SampleBuffer<f32>>,
    },
    Opus {
        decoder: Box<opuscule::Decoder>,
        head: OpusHead,
        pcm: Vec<f32>,
        /// Pre-skip samples still to drop.
        skip: usize,
    },
}

impl Codec {
    fn new(params: &CodecParameters) -> Result<(Self, u32)> {
        if params.codec == CODEC_TYPE_OPUS {
            let head = match params.extra_data.as_deref() {
                Some(head) => OpusHead::parse(head)?,
                // WebM may omit CodecPrivate; fall back to the track's channel count.
                None => OpusHead {
                    channels: params.channels.map_or(1, |c| c.count()).clamp(1, 2),
                    pre_skip: 0,
                    gain: 1.0,
                },
            };
            let channels = if head.channels == 1 {
                opuscule::Channels::Mono
            } else {
                opuscule::Channels::Stereo
            };
            let decoder = Box::new(opuscule::Decoder::new(
                opuscule::SampleRate::Hz48000,
                channels,
            ));
            let codec = Self::Opus {
                decoder,
                head,
                pcm: vec![0.0; OPUS_MAX_FRAMES * head.channels],
                skip: head.pre_skip,
            };
            return Ok((codec, OPUS_RATE));
        }
        let rate = params
            .sample_rate
            .ok_or_else(|| Error::InvalidInput("The audio track has no sample rate".into()))?;
        let decoder = symphonia::default::get_codecs()
            .make(params, &DecoderOptions::default())
            .map_err(|_| Error::InvalidInput(format!("Unsupported audio codec. {SUPPORTED}")))?;
        Ok((
            Self::Symphonia {
                decoder,
                buffer: None,
            },
            rate,
        ))
    }

    /// Appends the packet's samples, averaged to mono, to `mono`. Returns `false` for a corrupt
    /// packet, which is skipped as players do.
    fn decode(&mut self, packet: &Packet, mono: &mut Vec<f32>) -> Result<bool> {
        match self {
            Self::Symphonia { decoder, buffer } => {
                let audio = match decoder.decode(packet) {
                    Ok(audio) => audio,
                    Err(DecodeError::DecodeError(_)) => return Ok(false),
                    Err(_) => {
                        return Err(Error::InvalidInput(format!(
                            "Unrecognized or invalid audio. {SUPPORTED}"
                        )));
                    }
                };
                let spec = *audio.spec();
                let channels = spec.channels.count().max(1);
                let frames = audio.frames();
                let buffer = match buffer {
                    Some(b) if b.capacity() >= frames * channels => b,
                    slot => slot.insert(SampleBuffer::new(audio.capacity() as u64, spec)),
                };
                buffer.copy_interleaved_ref(audio);
                mono.extend(
                    buffer.samples()[..frames * channels]
                        .chunks_exact(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32),
                );
            }
            Self::Opus {
                decoder,
                head,
                pcm,
                skip,
            } => {
                let Ok(frames) = decoder.decode(Some(&packet.data), pcm, false) else {
                    return Ok(false);
                };
                let channels = head.channels;
                let dropped = (*skip).min(frames);
                *skip -= dropped;
                let scale = head.gain / channels as f32;
                mono.extend(
                    pcm[dropped * channels..frames * channels]
                        .chunks_exact(channels)
                        .map(|frame| frame.iter().sum::<f32>() * scale),
                );
            }
        }
        Ok(true)
    }
}

/// Upload bounds: compressed size and decoded duration.
#[derive(Clone, Copy, Debug)]
pub struct AudioLimits {
    pub max_bytes: usize,
    pub max_seconds: f64,
}

/// Decodes WAV, FLAC, MP3, OGG Vorbis, Ogg Opus, WebM (Opus) or M4A/MP4 (AAC) bytes to mono
/// f32 samples at `rate`.
/// `extension` (such as `"mp3"`, from the upload's file name) helps the container probe.
///
/// Channels are averaged; the audio is resampled while it decodes, so memory tracks the output.
/// Every failure is [`Error::InvalidInput`].
pub fn decode_audio(
    bytes: Vec<u8>,
    extension: Option<&str>,
    rate: u32,
    limits: AudioLimits,
) -> Result<Vec<f32>> {
    if bytes.is_empty() || bytes.len() > limits.max_bytes {
        return Err(Error::InvalidInput(format!(
            "Audio must contain 1 byte to {} MiB",
            limits.max_bytes / (1024 * 1024)
        )));
    }
    let invalid = |_| Error::InvalidInput(format!("Unrecognized or invalid audio. {SUPPORTED}"));
    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(invalid)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::InvalidInput("The audio file has no audio track".into()))?;
    let track_id = track.id;
    let (mut codec, source_rate) = Codec::new(&track.codec_params)?;

    let max_samples = (limits.max_seconds * f64::from(source_rate)) as usize;
    let too_long = || {
        Error::InvalidInput(format!(
            "Audio exceeds the {} second limit",
            limits.max_seconds
        ))
    };
    let mut resampler = Resampler::new(source_rate, rate);
    let mut output = Vec::new();
    let mut mono = Vec::new();
    let mut decoded = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(DecodeError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(DecodeError::ResetRequired) => break,
            Err(e) => return Err(invalid(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        mono.clear();
        if !codec.decode(&packet, &mut mono)? {
            continue;
        }
        decoded += mono.len();
        if decoded > max_samples {
            return Err(too_long());
        }
        resampler.process(&mono, &mut output);
    }
    resampler.finish(&mut output);
    if output.is_empty() {
        return Err(Error::InvalidInput(
            "The audio file contains no samples".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: AudioLimits = AudioLimits {
        max_bytes: 1024 * 1024,
        max_seconds: 30.0,
    };

    /// A 16-bit PCM WAV file with `channels` interleaved channels.
    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data = samples.len() as u32 * 2;
        let mut bytes = Vec::new();
        bytes.extend(b"RIFF");
        bytes.extend((36 + data).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(channels.to_le_bytes());
        bytes.extend(rate.to_le_bytes());
        bytes.extend((rate * u32::from(channels) * 2).to_le_bytes());
        bytes.extend((channels * 2).to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(data.to_le_bytes());
        for s in samples {
            bytes.extend(s.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn stereo_wav_is_averaged_to_mono_and_resampled() {
        // 0.5 s of stereo 8 kHz: left +8192, right -4096 → mono 2048 / 32768.
        let samples: Vec<i16> = (0..4000).flat_map(|_| [8192, -4096]).collect();
        let decoded = decode_audio(wav(8000, 2, &samples), Some("wav"), 16000, LIMITS).unwrap();
        assert_eq!(decoded.len(), 8000);
        let middle = decoded[4000];
        assert!((middle - 0.0625).abs() < 1e-3, "{middle}");
    }

    #[test]
    fn empty_oversized_long_and_garbage_audio_is_rejected() {
        let error = |bytes, limits| match decode_audio(bytes, None, 16000, limits) {
            Err(Error::InvalidInput(message)) => message,
            other => panic!("expected invalid input, got {other:?}"),
        };
        assert!(error(Vec::new(), LIMITS).contains("1 byte"));
        let big = AudioLimits {
            max_bytes: 10,
            ..LIMITS
        };
        assert!(error(wav(16000, 1, &[0; 100]), big).contains("MiB"));
        let short = AudioLimits {
            max_seconds: 1.0,
            ..LIMITS
        };
        assert!(error(wav(16000, 1, &vec![0; 32000]), short).contains("second limit"));
        assert!(error(b"not audio at all, just text".to_vec(), LIMITS).contains("Supported"));
        let mut truncated = example("speech-es.webm");
        truncated.truncate(40);
        assert!(error(truncated, LIMITS).contains("audio"));
    }

    #[test]
    #[ignore = "Requires the Parakeet reference dump (scripts/reference/parakeet_dump.py)"]
    fn example_flac_files_decode_like_the_reference_reader() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let golden = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join(".cache/jevons/golden/parakeet-tdt-0.6b-v3");
        for (clip, file) in [("en", "speech-en.flac"), ("es", "speech-es.flac")] {
            let bytes = std::fs::read(root.join("examples").join(file)).unwrap();
            let decoded = decode_audio(bytes, Some("flac"), 16000, LIMITS).unwrap();
            let reference: Vec<f32> = std::fs::read(golden.join(format!("{clip}_audio.f32")))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(decoded.len(), reference.len(), "{clip}");
            let max = decoded
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max < 1e-6, "{clip}: max |diff| {max}");
        }
    }

    fn example(file: &str) -> Vec<u8> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        std::fs::read(root.join(file)).unwrap()
    }

    /// Normalized cross-correlation of `a` and `b` over `range`, with `b` shifted by `shift`.
    fn correlation(a: &[f32], b: &[f32], shift: isize, range: std::ops::Range<usize>) -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for i in range {
            let Some(&y) = b.get((i as isize + shift) as usize) else {
                continue;
            };
            let x = f64::from(a[i]);
            ab += x * f64::from(y);
            aa += x * x;
            bb += f64::from(y).powi(2);
        }
        ab / (aa * bb).sqrt().max(f64::MIN_POSITIVE)
    }

    /// Correlation over the whole signal at the lag (within ±`lag`) that best aligns a
    /// one-second excerpt with speech in it.
    fn best_correlation(a: &[f32], b: &[f32], lag: isize) -> f64 {
        let n = a.len().min(b.len());
        let excerpt = 32000..48000;
        let score = |shift: isize| correlation(a, b, shift, excerpt.clone());
        let best = |shifts: &mut dyn Iterator<Item = isize>| {
            shifts
                .max_by(|&x, &y| score(x).total_cmp(&score(y)))
                .unwrap()
        };
        // Coarse steps, then the neighbours of the best one.
        let coarse = best(&mut (-lag..=lag).step_by(4));
        let shift = best(&mut (coarse - 3..=coarse + 3));
        correlation(a, b, shift, lag as usize..n - lag as usize)
    }

    #[test]
    fn webm_and_ogg_opus_decode_like_their_lossless_originals() {
        for (opus, flac, extension) in [
            ("speech-es.webm", "speech-es.flac", "webm"),
            // Chrome's MediaRecorder: stereo 60 ms packets, no duration, 250 ms timeslices.
            ("speech-es-browser.webm", "speech-es.flac", "webm"),
            ("speech-en.opus", "speech-en.flac", "opus"),
        ] {
            let original = decode_audio(example(flac), Some("flac"), 16000, LIMITS).unwrap();
            // The container is found without the file name's help, too.
            for hint in [Some(extension), None] {
                let decoded = decode_audio(example(opus), hint, 16000, LIMITS).unwrap();
                let extra = decoded.len() as f64 / original.len() as f64 - 1.0;
                assert!(
                    extra.abs() < 0.01,
                    "{opus}: {} vs {} samples",
                    decoded.len(),
                    original.len()
                );
                // Browsers start recording a little before playback; allow 0.3 s.
                let correlation = best_correlation(&original, &decoded, 4800);
                assert!(correlation > 0.9, "{opus}: correlation {correlation}");
            }
        }
    }

    #[test]
    fn opus_heads_give_channels_pre_skip_and_gain_and_reject_surround() {
        let mut head = b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x06\x00".to_vec();
        let parsed = OpusHead::parse(&head).unwrap();
        assert_eq!((parsed.channels, parsed.pre_skip), (2, 312));
        // +6 dB in Q7.8 is 1536.
        assert!((parsed.gain - 10f32.powf(6.0 / 20.0)).abs() < 1e-6);
        head[9] = 6;
        head[18] = 1;
        assert!(OpusHead::parse(&head).is_err());
        assert!(OpusHead::parse(b"OpusTags\x01\x01\x00\x00\x80\xbb\x00\x00\x00\x00\x00").is_err());
        assert!(OpusHead::parse(b"OpusHead").is_err());
    }

    #[test]
    #[ignore = "Requires a libopus reference: ffmpeg -c:a libopus -i examples/speech-es.webm -f f32le -ac 1 -ar 48000 $GOLDEN/es_webm_libopus48k.f32"]
    fn webm_opus_decodes_bit_exactly_like_libopus() {
        let golden = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join(".cache/jevons/golden/parakeet-tdt-0.6b-v3/es_webm_libopus48k.f32");
        let reference: Vec<f32> = std::fs::read(golden)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let decoded = decode_audio(example("speech-es.webm"), None, 48000, LIMITS).unwrap();
        // libopus also trims the end padding by the final granule position; compare the rest.
        assert!(decoded.len() >= reference.len());
        let worst = decoded
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "max |diff| {worst}");
    }
}
