//! GGML block formats: CPU reference dequantization and GPU-aligned repacking.
//!
//! References: pinned `ggml-quants.c` (`dequantize_row_q4_K`, `dequantize_row_q6_K`,
//! `dequantize_row_q5_0`, `dequantize_row_q8_0`). Repacking changes only byte placement so every
//! block starts on a 4-byte boundary; decoded values are identical to the source blocks.
use crate::gguf::TensorType;
use half::f16;

fn half(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

/// Q4_K 6-bit scale/min pair `j` from the 12 packed scale bytes.
pub fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequant_q4k(block: &[u8], out: &mut [f32]) {
    let d = half(&block[0..]);
    let dmin = half(&block[2..]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    for pair in 0..4 {
        let (s0, m0) = q4k_scale_min(2 * pair, scales);
        let (s1, m1) = q4k_scale_min(2 * pair + 1, scales);
        let (d0, n0) = (d * f32::from(s0), dmin * f32::from(m0));
        let (d1, n1) = (d * f32::from(s1), dmin * f32::from(m1));
        for l in 0..32 {
            let q = qs[pair * 32 + l];
            out[pair * 64 + l] = d0 * f32::from(q & 15) - n0;
            out[pair * 64 + 32 + l] = d1 * f32::from(q >> 4) - n1;
        }
    }
}

fn dequant_q6k(block: &[u8], out: &mut [f32]) {
    let (ql, qh, sc) = (&block[0..128], &block[128..192], &block[192..208]);
    let d = half(&block[208..]);
    for n in 0..2 {
        let (ql, qh, sc) = (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..]);
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            let o = &mut out[128 * n..];
            o[l] = d * f32::from(sc[is] as i8) * q1 as f32;
            o[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
            o[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
            o[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
        }
    }
}

fn dequant_q5_0(block: &[u8], out: &mut [f32]) {
    let d = half(block);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    for j in 0..16 {
        let h0 = ((qh >> j) << 4) & 0x10;
        let h1 = (qh >> (j + 12)) & 0x10;
        out[j] = d * ((u32::from(qs[j] & 15) | h0) as i32 - 16) as f32;
        out[j + 16] = d * ((u32::from(qs[j] >> 4) | h1) as i32 - 16) as f32;
    }
}

fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    let d = half(block);
    for (o, &q) in out.iter_mut().zip(&block[2..34]) {
        *o = d * f32::from(q as i8);
    }
}

/// Dequantizes whole rows of a supported tensor type.
pub fn dequantize(kind: TensorType, bytes: &[u8], out: &mut [f32]) -> Result<(), String> {
    let (block, size) = kind.block().ok_or("unsupported tensor type")?;
    let (block, size) = (block as usize, size as usize);
    if !bytes.len().is_multiple_of(size) || out.len() != bytes.len() / size * block {
        return Err("dequantization size mismatch".into());
    }
    for (src, dst) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(block)) {
        match kind {
            TensorType::F32 => dst[0] = f32::from_le_bytes([src[0], src[1], src[2], src[3]]),
            TensorType::F16 => dst[0] = half(src),
            TensorType::Q4K => dequant_q4k(src, dst),
            TensorType::Q6K => dequant_q6k(src, dst),
            TensorType::Q5_0 => dequant_q5_0(src, dst),
            TensorType::Q8_0 => dequant_q8_0(src, dst),
            TensorType::Other(_) => unreachable!(),
        }
    }
    Ok(())
}

/// Structure-of-arrays GPU layout: every region is 16-byte aligned per block so kernels can use
/// vector loads. Values decode exactly as the GGML source blocks.
///
/// * Q4_K: `q` = original 144-byte blocks (36 words); other regions empty.
/// * Q6_K: `q` = `ql` (32 words/block), `h` = `qh` (16 words/block), `s` = int8 scales
///   (4 words/block), `d` = f32 scale.
/// * Q5_0: `q` = `qs` (4 words/block), `h` = high bits (1 word/block), `d` = f32 scale.
/// * Q8_0: `q` = int8 quants (8 words/block), `d` = f32 scale.
#[derive(Debug, Default)]
pub struct Packed {
    pub q: Vec<u32>,
    pub h: Vec<u32>,
    pub s: Vec<u32>,
    pub d: Vec<f32>,
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
}

/// Words per block in each packed region `(q, h, s, d)` (see [`Packed`]).
pub fn packed_words(kind: TensorType) -> (usize, usize, usize, usize) {
    match kind {
        TensorType::Q4K => (36, 0, 0, 0),
        TensorType::Q6K => (32, 16, 4, 1),
        TensorType::Q5_0 => (4, 1, 0, 1),
        TensorType::Q8_0 => (8, 0, 0, 1),
        _ => (0, 0, 0, 0),
    }
}

impl Packed {
    /// Appends another tensor's blocks (row concatenation of matrices with equal `k`).
    pub fn extend(&mut self, other: Packed) {
        self.q.extend(other.q);
        self.h.extend(other.h);
        self.s.extend(other.s);
        self.d.extend(other.d);
    }
}

/// Whether [`pack`] has a GPU layout for `kind`.
pub fn packed_supported(kind: TensorType) -> bool {
    matches!(
        kind,
        TensorType::Q4K | TensorType::Q6K | TensorType::Q5_0 | TensorType::Q8_0
    )
}

pub fn pack(kind: TensorType, bytes: &[u8]) -> Result<Packed, String> {
    let (_, size) = kind.block().ok_or("unsupported tensor type")?;
    let size = size as usize;
    if !bytes.len().is_multiple_of(size) {
        return Err("tensor is not a whole number of blocks".into());
    }
    let blocks = bytes.len() / size;
    let mut p = Packed::default();
    match kind {
        TensorType::Q4K => p.q.extend(words(bytes)),
        TensorType::Q6K => {
            p.q.reserve(blocks * 32);
            p.h.reserve(blocks * 16);
            p.s.reserve(blocks * 4);
            for b in bytes.chunks_exact(size) {
                p.q.extend(words(&b[..128]));
                p.h.extend(words(&b[128..192]));
                p.s.extend(words(&b[192..208]));
                p.d.push(half(&b[208..]));
            }
        }
        TensorType::Q5_0 => {
            for b in bytes.chunks_exact(size) {
                p.d.push(half(b));
                p.h.extend(words(&b[2..6]));
                p.q.extend(words(&b[6..22]));
            }
        }
        TensorType::Q8_0 => {
            for b in bytes.chunks_exact(size) {
                p.d.push(half(b));
                p.q.extend(words(&b[2..34]));
            }
        }
        _ => return Err(format!("{kind:?} has no GPU block layout")),
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_decodes_scale_high_bits_for_upper_groups() {
        let q = [
            0xc1, 0x82, 0x43, 0x04, 0x85, 0xc6, 0x07, 0x48, 0x12, 0x34, 0x56, 0x78,
        ];
        let pairs: Vec<_> = (0..8).map(|j| q4k_scale_min(j, &q)).collect();
        assert_eq!(pairs[0], (1, 5));
        assert_eq!(pairs[4], (50, 33));
        assert_eq!(pairs[7], (8, 23));
    }

    #[test]
    fn q8_0_and_q5_0_dequantize_signed_values() {
        let mut q8 = vec![0u8; 34];
        q8[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        q8[2] = (-4i8) as u8;
        q8[33] = 7;
        let mut out = vec![0.0; 32];
        dequantize(TensorType::Q8_0, &q8, &mut out).unwrap();
        assert_eq!((out[0], out[31]), (-2.0, 3.5));

        let mut q5 = vec![0u8; 22];
        q5[..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        q5[2..6].copy_from_slice(&(1u32 | (1 << 16)).to_le_bytes());
        q5[6] = 0x3f;
        dequantize(TensorType::Q5_0, &q5, &mut out).unwrap();
        assert_eq!((out[0], out[16], out[1]), (15.0, 3.0, -16.0));
    }

    #[test]
    fn packing_splits_quants_and_scales_without_changing_values() {
        let mut q8 = vec![0u8; 34];
        q8[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        q8[2..6].copy_from_slice(&[1, 2, 3, 4]);
        let p = pack(TensorType::Q8_0, &q8).unwrap();
        assert_eq!((p.q.len(), p.d.as_slice()), (8, &[0.5][..]));
        assert_eq!(p.q[0], u32::from_le_bytes([1, 2, 3, 4]));
        assert!(pack(TensorType::Q8_0, &q8[..33]).is_err());
    }
}
