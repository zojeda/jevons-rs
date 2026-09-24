//! Streams checkpoint tensors to the device one at a time.
use burn::tensor::{DType, Device, Tensor, TensorData};
use jevons_formats::safetensors::{Checkpoint, Dtype, SafetensorsError};

#[derive(Debug)]
pub enum WeightError {
    Read(SafetensorsError),
    Shape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    Dtype(String),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(e) => write!(f, "{e}"),
            Self::Shape {
                name,
                expected,
                found,
            } => write!(f, "{name}: expected shape {expected:?}, found {found:?}"),
            Self::Dtype(name) => write!(f, "{name}: expected BF16 weights"),
        }
    }
}

impl From<SafetensorsError> for WeightError {
    fn from(error: SafetensorsError) -> Self {
        Self::Read(error)
    }
}

pub struct Loader<'a> {
    pub checkpoint: &'a Checkpoint,
    pub device: &'a Device,
}

impl Loader<'_> {
    fn bytes(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<u8>, WeightError> {
        let info = self.checkpoint.tensor(name)?;
        let expected = if cols == 0 {
            vec![rows]
        } else {
            vec![rows, cols]
        };
        if info.shape != expected {
            return Err(WeightError::Shape {
                name: name.into(),
                expected,
                found: info.shape.clone(),
            });
        }
        if info.dtype != Dtype::BF16 {
            return Err(WeightError::Dtype(name.into()));
        }
        let mut bytes = vec![0; info.byte_len()];
        self.checkpoint.read_into(info, &mut bytes)?;
        Ok(bytes)
    }

    fn persistent<const D: usize>(&self, data: TensorData, dtype: DType) -> Tensor<D> {
        self.device
            .memory_persistent_allocations(data, |d| Tensor::from_data(d, (self.device, dtype)))
    }

    /// A BF16 matrix `[rows, cols]`, kept BF16 on the device.
    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Tensor<2>, WeightError> {
        self.stacked(&[(name, rows)], cols)
    }

    /// BF16 matrices with a shared column count stacked by rows, such as fused Q/K/V.
    pub fn stacked(&self, parts: &[(&str, usize)], cols: usize) -> Result<Tensor<2>, WeightError> {
        let rows: usize = parts.iter().map(|(_, r)| r).sum();
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for (name, r) in parts {
            bytes.extend(self.bytes(name, *r, cols)?);
        }
        let data = TensorData::from_bytes_vec(bytes, [rows, cols], DType::BF16);
        Ok(self.persistent(data, DType::BF16))
    }

    /// BF16 matrices stacked by rows and converted to FP16 for the tuned GEMM, with zero rows
    /// appended up to a multiple of `row_multiple`.
    pub fn stacked_f16(
        &self,
        parts: &[(&str, usize)],
        cols: usize,
        row_multiple: usize,
    ) -> Result<Tensor<2>, WeightError> {
        let rows: usize = parts.iter().map(|(_, r)| r).sum();
        let padded = rows.next_multiple_of(row_multiple);
        let mut halves = Vec::with_capacity(padded * cols * 2);
        for (name, r) in parts {
            let bytes = self.bytes(name, *r, cols)?;
            halves.extend(bytes.as_chunks::<2>().0.iter().flat_map(|c| {
                half::f16::from_f32(half::bf16::from_le_bytes(*c).to_f32()).to_le_bytes()
            }));
        }
        halves.resize(padded * cols * 2, 0);
        let data = TensorData::from_bytes_vec(halves, [padded, cols], DType::F16);
        Ok(self.persistent(data, DType::F16))
    }

    /// A BF16 tensor of any rank whose first dimension is `rows`, flattened to `[rows, cols]`
    /// and widened to f32 (such as a convolution kernel used as a matrix).
    pub fn reshaped_f32(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor<2>, WeightError> {
        let info = self.checkpoint.tensor(name)?;
        if info.shape.first() != Some(&rows) || info.elements() != rows * cols {
            return Err(WeightError::Shape {
                name: name.into(),
                expected: vec![rows, cols],
                found: info.shape.clone(),
            });
        }
        if info.dtype != Dtype::BF16 {
            return Err(WeightError::Dtype(name.into()));
        }
        let mut bytes = vec![0; info.byte_len()];
        self.checkpoint.read_into(info, &mut bytes)?;
        let values: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect();
        Ok(self.persistent(TensorData::new(values, [rows, cols]), DType::F32))
    }

    /// A BF16 vector widened to f32, such as a norm weight.
    pub fn vector_f32(&self, name: &str, len: usize) -> Result<Tensor<1>, WeightError> {
        let bytes = self.bytes(name, len, 0)?;
        let values: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect();
        Ok(self.persistent(TensorData::new(values, [len]), DType::F32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    #[ignore = "Requires NEMOTRON_MODEL and a HIP GPU"]
    fn fp16_conversion_preserves_real_projection_weights() {
        let dir = std::path::PathBuf::from(std::env::var("NEMOTRON_MODEL").unwrap());
        let checkpoint = Checkpoint::open_dir(&dir).unwrap();
        let device = crate::device::hip(0);
        let load = Loader {
            checkpoint: &checkpoint,
            device: &device,
        };
        for (name, rows, cols) in [
            (
                "encoder.layers.0.self_attn.q_proj.weight",
                4096usize,
                4096usize,
            ),
            ("encoder.layers.0.mlp.down_proj.weight", 4096, 14336),
        ] {
            let bf16 = load.matrix(name, rows, cols).unwrap();
            let f16 = load.stacked_f16(&[(name, rows)], cols, 64).unwrap();
            let x = Tensor::<2>::random(
                [4, cols],
                Distribution::Uniform(-1.0, 1.0),
                (&device, DType::F32),
            );
            let a: Vec<f32> = x
                .clone()
                .matmul(bf16.cast(DType::F32).transpose())
                .into_data()
                .try_to_vec()
                .unwrap();
            let b: Vec<f32> = x
                .matmul(f16.cast(DType::F32).transpose())
                .into_data()
                .try_to_vec()
                .unwrap();
            let worst = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f32::max);
            let scale = a.iter().map(|v| v.abs()).fold(0.0, f32::max);
            println!("{name}: max difference {worst} of {scale}");
            assert!(worst <= 1e-3 * scale, "{name}");
        }
    }
}
