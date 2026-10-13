//! Mono 16-bit PCM WAV output: the header alone for a stream, or a whole file.
use crate::pcm::pcm16_le_bytes;

/// What a header writes in a size field when the length is unknown.
const UNKNOWN_SIZE: u32 = u32::MAX;
/// Bytes of the header after the RIFF size field.
const HEADER_REST: u32 = 36;

/// The 44-byte RIFF/WAVE header of a mono 16-bit PCM file at `sample_rate`. `Some(n)` describes
/// a file of `n` samples with its true sizes. `None` describes a stream of unknown length: the
/// RIFF and data sizes are `0xFFFFFFFF`, which players read as "until the data ends". A file
/// too long for the 32-bit size fields gets the same sizes.
pub fn wav_header(sample_rate: u32, samples: Option<u32>) -> [u8; 44] {
    let sizes = samples
        .and_then(|samples| samples.checked_mul(2))
        .and_then(|data| Some((data.checked_add(HEADER_REST)?, data)));
    let (riff, data) = sizes.unwrap_or((UNKNOWN_SIZE, UNKNOWN_SIZE));
    let mut header = [0; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&riff.to_le_bytes());
    header[8..16].copy_from_slice(b"WAVEfmt ");
    // The format chunk: 16 bytes, PCM, one channel.
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes());
    header[22..24].copy_from_slice(&1u16.to_le_bytes());
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    // Bytes per second, bytes per frame, bits per sample.
    header[28..32].copy_from_slice(&sample_rate.saturating_mul(2).to_le_bytes());
    header[32..34].copy_from_slice(&2u16.to_le_bytes());
    header[34..36].copy_from_slice(&16u16.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data.to_le_bytes());
    header
}

/// A whole mono 16-bit PCM WAV file: [`wav_header`] with the true sizes, then `samples`
/// through [`pcm16_le_bytes`].
pub fn wav(sample_rate: u32, samples: &[f32]) -> Vec<u8> {
    let count = u32::try_from(samples.len()).ok();
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend(wav_header(sample_rate, count));
    bytes.extend(pcm16_le_bytes(samples));
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioLimits, decode_audio};

    const LIMITS: AudioLimits = AudioLimits {
        max_bytes: 1024 * 1024,
        max_seconds: 30.0,
    };

    fn tone(rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate);
                0.5 * phase.sin() as f32
            })
            .collect()
    }

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
    }

    #[test]
    fn a_wav_header_describes_mono_pcm16_with_its_true_sizes() {
        let header = wav_header(24000, Some(100));
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(u32_at(&header, 4), 36 + 200);
        assert_eq!(&header[8..16], b"WAVEfmt ");
        assert_eq!(u32_at(&header, 16), 16);
        // PCM, one channel.
        assert_eq!((u16_at(&header, 20), u16_at(&header, 22)), (1, 1));
        assert_eq!(u32_at(&header, 24), 24000);
        assert_eq!(u32_at(&header, 28), 48000);
        // Two bytes a frame, 16 bits a sample.
        assert_eq!((u16_at(&header, 32), u16_at(&header, 34)), (2, 16));
        assert_eq!(&header[36..40], b"data");
        assert_eq!(u32_at(&header, 40), 200);
        // An empty file is a header alone.
        assert_eq!(u32_at(&wav_header(16000, Some(0)), 4), 36);
        assert_eq!(u32_at(&wav_header(16000, Some(0)), 40), 0);
    }

    #[test]
    fn a_streamed_wav_header_marks_both_sizes_unknown() {
        let streamed = wav_header(24000, None);
        assert_eq!(u32_at(&streamed, 4), 0xFFFF_FFFF);
        assert_eq!(u32_at(&streamed, 40), 0xFFFF_FFFF);
        // Nothing else differs from a file's header.
        let file = wav_header(24000, Some(100));
        assert_eq!(streamed[8..40], file[8..40]);
        // A length the 32-bit fields cannot hold reads as unknown too.
        assert_eq!(wav_header(24000, Some(u32::MAX)), streamed);
        assert_eq!(wav_header(24000, Some(0x7FFF_FFFF)), streamed);
        assert_eq!(
            u32_at(&wav_header(24000, Some(0x7FFF_FFED)), 4),
            0xFFFF_FFFE
        );
    }

    #[test]
    fn a_written_wav_is_its_header_then_pcm16_and_decodes_back_to_its_samples() {
        let samples = tone(24000, 2400);
        let file = wav(24000, &samples);
        assert_eq!(file[..44], wav_header(24000, Some(2400)));
        assert_eq!(file[44..], pcm16_le_bytes(&samples));
        let decoded = decode_audio(file, Some("wav"), 24000, LIMITS).unwrap();
        assert_eq!(decoded.len(), samples.len());
        for (read, written) in decoded.iter().zip(&samples) {
            assert!((read - written).abs() < 1e-4, "{read} {written}");
        }
    }

    #[test]
    fn a_streamed_wav_decodes_to_the_end_of_its_data() {
        let samples = tone(24000, 2400);
        let mut stream = wav_header(24000, None).to_vec();
        stream.extend(pcm16_le_bytes(&samples));
        let decoded = decode_audio(stream, Some("wav"), 24000, LIMITS).unwrap();
        assert_eq!(decoded.len(), samples.len());
        for (read, written) in decoded.iter().zip(&samples) {
            assert!((read - written).abs() < 1e-4, "{read} {written}");
        }
    }
}
