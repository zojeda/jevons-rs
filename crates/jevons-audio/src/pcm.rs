//! Raw sample formats of streaming audio: 16-bit little-endian PCM and G.711 µ-law / A-law.

/// 16-bit little-endian PCM to f32 in `[-1, 1)`. A trailing odd byte is ignored.
pub fn pcm16_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect()
}

/// G.711 µ-law bytes to f32.
pub fn mulaw(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&byte| {
            let u = !byte;
            let magnitude = ((i32::from(u & 0x0f) << 3) + 0x84) << ((u >> 4) & 0x07);
            let value = if u & 0x80 != 0 {
                0x84 - magnitude
            } else {
                magnitude - 0x84
            };
            value as f32 / 32768.0
        })
        .collect()
}

/// G.711 A-law bytes to f32.
pub fn alaw(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&byte| {
            let a = byte ^ 0x55;
            let exponent = (a >> 4) & 0x07;
            let mantissa = i32::from(a & 0x0f);
            let magnitude = match exponent {
                0 => (mantissa << 4) + 8,
                e => ((mantissa << 4) + 0x108) << (e - 1),
            };
            let value = if a & 0x80 != 0 { magnitude } else { -magnitude };
            value as f32 / 32768.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_covers_the_full_range_and_ignores_a_trailing_byte() {
        let bytes = [0x00, 0x80, 0xff, 0x7f, 0x00, 0x00, 0x01];
        assert_eq!(pcm16_le(&bytes), vec![-1.0, 32767.0 / 32768.0, 0.0]);
    }

    #[test]
    fn g711_decodes_reference_codes() {
        // ITU-T G.711 reference points: silence, and the largest positive and negative codes.
        assert_eq!(mulaw(&[0xff, 0x7f]), vec![0.0, 0.0]);
        assert_eq!(mulaw(&[0x80])[0], 32124.0 / 32768.0);
        assert_eq!(mulaw(&[0x00])[0], -32124.0 / 32768.0);
        assert_eq!(alaw(&[0xd5])[0], 8.0 / 32768.0);
        assert_eq!(alaw(&[0x55])[0], -8.0 / 32768.0);
        assert_eq!(alaw(&[0xaa])[0], 32256.0 / 32768.0);
        assert_eq!(alaw(&[0x2a])[0], -32256.0 / 32768.0);
    }
}
