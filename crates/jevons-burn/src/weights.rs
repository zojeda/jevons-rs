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
