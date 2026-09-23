//! Q4_K weight preparation and GPU dequantization using safe Burn operations.
//!
//! Reference layout: pinned ggml-quants.c dequantize_row_q4_K. Scale/min metadata
//! is expanded on the CPU once at load; 4-bit values remain packed on the device.
//! This path materializes float weights, not a fused quantized matmul kernel.
use burn_rocm::{Rocm, RocmDevice};
use burn_tensor::{DType, Int, Tensor, TensorData};
use half::f16;

pub type HipBackend = Rocm<f32>;

pub struct Q4kWeight {
    quants: Tensor<HipBackend, 3, Int>,
    scales: Tensor<HipBackend, 3>,
    minima: Tensor<HipBackend, 3>,
    rows: usize,
    columns: usize,
}

pub(crate) fn block_scales(block: &[u8; 144]) -> ([f32; 8], [f32; 8]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let q = &block[4..16];
    let mut scales = [0.0; 8];
    let mut minima = [0.0; 8];
    for j in 0..8 {
        let (scale, min) = if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        };
        scales[j] = d * f32::from(scale);
        minima[j] = dmin * f32::from(min);
    }
    (scales, minima)
}

impl Q4kWeight {
    pub fn upload(
        bytes: &[u8],
        rows: usize,
        columns: usize,
        device: &RocmDevice,
    ) -> Result<Self, String> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(256) {
            return Err("Q4_K requires nonzero rows and a column count divisible by 256".into());
        }
        let blocks = rows
            .checked_mul(columns / 256)
            .ok_or("Q4_K size overflow")?;
        if blocks.checked_mul(144) != Some(bytes.len()) {
            return Err("Q4_K byte count does not match the matrix shape".into());
        }
        let mut quants = Vec::with_capacity(blocks * 128);
        let mut scales = Vec::with_capacity(blocks * 8);
        let mut minima = Vec::with_capacity(blocks * 8);
        for block in bytes.as_chunks::<144>().0 {
            let (d, m) = block_scales(block);
            if d.iter().chain(&m).any(|v| !v.is_finite()) {
                return Err("Q4_K scale metadata is nonfinite".into());
            }
            scales.extend(d);
            minima.extend(m);
            quants.extend_from_slice(&block[16..]);
        }
        Ok(Self {
            quants: Tensor::from_data(
                TensorData::new(quants, [blocks, 4, 32]),
                (device, DType::U8),
            ),
            scales: Tensor::from_data(TensorData::new(scales, [blocks, 8, 1]), device),
            minima: Tensor::from_data(TensorData::new(minima, [blocks, 8, 1]), device),
            rows,
            columns,
        })
    }

    /// Device tensor payload only; runtime allocations and scratch are additional.
    pub fn prepared_bytes(&self) -> usize {
        self.rows * self.columns / 256 * (128 * self.quants.dtype().size() + 8 * 4 * 2)
    }

    pub fn dequantize(&self) -> Tensor<HipBackend, 2> {
        let lo = self.quants.clone().bitwise_and_scalar(15);
        let hi = self.quants.clone().bitwise_right_shift_scalar(4);
        let blocks = self.rows * self.columns / 256;
        let values = Tensor::stack::<4>(vec![lo, hi], 2)
            .reshape([blocks, 8, 32])
            .float();
        (values * self.scales.clone() - self.minima.clone()).reshape([self.rows, self.columns])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_weights_fail_before_touching_the_device() {
        let device = RocmDevice::new(0);
        for (rows, columns, bytes) in [
            (0, 256, vec![]),
            (1, 255, vec![0; 144]),
            (1, 256, vec![0; 143]),
            (usize::MAX, 512, vec![]),
        ] {
            assert!(Q4kWeight::upload(&bytes, rows, columns, &device).is_err());
        }
        let mut block = [0u8; 144];
        block[..2].copy_from_slice(&f16::INFINITY.to_le_bytes());
        assert!(Q4kWeight::upload(&block, 1, 256, &device).is_err());
    }

    #[test]
    fn q4k_scales_preserve_the_high_bits_for_both_halves() {
        let mut block = [0u8; 144];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
        block[4..16].copy_from_slice(&[
            0xc1, 0x82, 0x43, 0x04, 0x85, 0xc6, 0x07, 0x48, 0x12, 0x34, 0x56, 0x78,
        ]);
        let (d, m) = block_scales(&block);
        assert_eq!(d, [0.5, 1.0, 1.5, 2.0, 25.0, 18.0, 11.0, 4.0]);
        assert_eq!(m, [1.25, 1.5, 1.75, 2.0, 8.25, 12.75, 1.25, 5.75]);
    }
}
