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
            Self::Dtype(name) => write!(f, "{name}: unsupported weight dtype"),
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
    /// Raw bytes of `name`, which must have exactly `shape`.
    fn raw(&self, name: &str, shape: &[usize]) -> Result<(Dtype, Vec<u8>), WeightError> {
        let info = self.checkpoint.tensor(name)?;
        if info.shape != shape {
            return Err(WeightError::Shape {
                name: name.into(),
                expected: shape.to_vec(),
                found: info.shape.clone(),
            });
        }
        let mut bytes = vec![0; info.byte_len()];
        self.checkpoint.read_into(info, &mut bytes)?;
        Ok((info.dtype, bytes))
    }

    fn bytes(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<u8>, WeightError> {
        let shape = if cols == 0 {
            vec![rows]
        } else {
            vec![rows, cols]
        };
        match self.raw(name, &shape)? {
            (Dtype::BF16, bytes) => Ok(bytes),
            _ => Err(WeightError::Dtype(name.into())),
        }
    }

    /// A BF16, FP16 or F32 tensor of exactly `shape`, widened to f32 on the host.
    pub fn host_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, WeightError> {
        let (dtype, bytes) = self.raw(name, shape)?;
        widen(dtype, &bytes).ok_or_else(|| WeightError::Dtype(name.into()))
    }

    /// Host f32 values uploaded as an f32 tensor.
    pub fn upload_f32<const D: usize>(&self, values: Vec<f32>, shape: [usize; D]) -> Tensor<D> {
        self.persistent(TensorData::new(values, shape), DType::F32)
    }

    /// A BF16, FP16 or F32 tensor of exactly `shape`, as f32 on the device.
    pub fn tensor_f32<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
    ) -> Result<Tensor<D>, WeightError> {
        Ok(self.upload_f32(self.host_f32(name, &shape)?, shape))
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

    /// BF16, FP16 or F32 matrices stacked by rows and converted to FP16 for the tuned GEMM,
    /// with zero rows appended up to a multiple of `row_multiple`.
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
            match self.raw(name, &[*r, cols])? {
                (Dtype::BF16, bytes) => {
                    halves.extend(bytes.as_chunks::<2>().0.iter().flat_map(|c| {
                        half::f16::from_f32(half::bf16::from_le_bytes(*c).to_f32()).to_le_bytes()
                    }))
                }
                (Dtype::F16, bytes) => halves.extend(bytes),
                (Dtype::F32, bytes) => {
                    halves.extend(
                        bytes.as_chunks::<4>().0.iter().flat_map(|c| {
                            half::f16::from_f32(f32::from_le_bytes(*c)).to_le_bytes()
                        }),
                    )
                }
                _ => return Err(WeightError::Dtype((*name).into())),
            }
        }
        halves.resize(padded * cols * 2, 0);
        let data = TensorData::from_bytes_vec(halves, [padded, cols], DType::F16);
        Ok(self.persistent(data, DType::F16))
    }

    /// Host f32 matrix values `[rows, cols]` converted to FP16 for the tuned GEMM, with zero
    /// rows appended up to a multiple of `row_multiple`.
    pub fn upload_f16(
        &self,
        values: &[f32],
        rows: usize,
        cols: usize,
        row_multiple: usize,
    ) -> Tensor<2> {
        assert_eq!(values.len(), rows * cols, "matrix values");
        let padded = rows.next_multiple_of(row_multiple);
        let mut halves: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        halves.resize(padded * cols * 2, 0);
        let data = TensorData::from_bytes_vec(halves, [padded, cols], DType::F16);
        self.persistent(data, DType::F16)
    }

    /// A BF16, FP16 or F32 tensor of any rank whose first dimension is `rows`, flattened to
    /// `[rows, cols]` and widened to f32 (such as a convolution kernel used as a matrix).
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
        let values = self.host_f32(name, &info.shape.clone())?;
        Ok(self.upload_f32(values, [rows, cols]))
    }

    /// A BF16, FP16 or F32 vector widened to f32, such as a norm weight.
    pub fn vector_f32(&self, name: &str, len: usize) -> Result<Tensor<1>, WeightError> {
        self.tensor_f32(name, [len])
    }
}

/// Little-endian BF16, FP16 or F32 values as f32.
fn widen(dtype: Dtype, bytes: &[u8]) -> Option<Vec<f32>> {
    Some(match dtype {
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        _ => return None,
    })
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
