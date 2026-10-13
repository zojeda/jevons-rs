//! Weights as the runtime wants them: FP16 for the tuned GEMM on a CubeCL device, f32 on the
//! CPU device the parity tests run on.
use jevons_burn::layers::{linear, linear_unbounded};
use jevons_burn::weights::{Loader, WeightError};
use jevons_burn::{DType, Tensor};

/// The tuned GEMM needs row and column counts that are multiples of this.
const TILE: usize = 64;

pub(crate) struct Weights<'a> {
    pub load: Loader<'a>,
    /// Whether matrices go to the device as FP16 for the tuned GEMM.
    pub tuned: bool,
}

/// A matrix `[outputs, inputs]` applied to rows. On a CubeCL device it is padded with zeros
/// to the GEMM's tile, and [`Matrix::apply`] pads its input and cuts its output to match.
pub(crate) struct Matrix {
    weight: Tensor<2>,
    outputs: usize,
}

impl Matrix {
    /// `x @ weightᵀ` for rows bounded to the FP16 range.
    pub fn apply(&self, x: Tensor<2>) -> Tensor<2> {
        self.product(x, linear)
    }

    /// [`Matrix::apply`] for rows that may exceed the FP16 range.
    pub fn apply_unbounded(&self, x: Tensor<2>) -> Tensor<2> {
        self.product(x, linear_unbounded)
    }

    fn product(&self, x: Tensor<2>, f: fn(Tensor<2>, &Tensor<2>) -> Tensor<2>) -> Tensor<2> {
        let [rows, cols] = x.dims();
        let [padded_outputs, inputs] = self.weight.dims();
        let x = if inputs > cols {
            let zeros = Tensor::zeros([rows, inputs - cols], (&x.device(), DType::F32));
            Tensor::cat(vec![x, zeros], 1)
        } else {
            x
        };
        let y = f(x, &self.weight);
        if padded_outputs > self.outputs {
            y.slice([0..rows, 0..self.outputs])
        } else {
            y
        }
    }
}

impl Weights<'_> {
    /// Checkpoint matrices with a shared column count, stacked by rows (such as Q, K and V).
    /// Their sizes must be multiples of the GEMM's tile.
    pub fn stacked(&self, parts: &[(&str, usize)], cols: usize) -> Result<Tensor<2>, WeightError> {
        if self.tuned {
            return self.load.stacked_f16(parts, cols, TILE);
        }
        let rows: usize = parts.iter().map(|(_, r)| r).sum();
        let mut values = Vec::with_capacity(rows * cols);
        for (name, r) in parts {
            values.extend(self.load.host_f32(name, &[*r, cols])?);
        }
        Ok(self.load.upload_f32(values, [rows, cols]))
    }

    /// A matrix built on the host, `[outputs, inputs]` row-major, of any size.
    pub fn matrix(&self, values: &[f32], outputs: usize, inputs: usize) -> Matrix {
        assert_eq!(values.len(), outputs * inputs, "matrix values");
        if !self.tuned {
            return Matrix {
                weight: self.load.upload_f32(values.to_vec(), [outputs, inputs]),
                outputs,
            };
        }
        let padded = inputs.next_multiple_of(TILE);
        let weight = if padded == inputs {
            self.load.upload_f16(values, outputs, inputs, TILE)
        } else {
            let mut wide = vec![0.0; outputs * padded];
            for (row, source) in wide
                .chunks_exact_mut(padded)
                .zip(values.chunks_exact(inputs))
            {
                row[..inputs].copy_from_slice(source);
            }
            self.load.upload_f16(&wide, outputs, padded, TILE)
        };
        Matrix { weight, outputs }
    }

    /// A checkpoint matrix `[outputs, inputs]` of any size.
    pub fn named(&self, name: &str, outputs: usize, inputs: usize) -> Result<Matrix, WeightError> {
        Ok(self.matrix(
            &self.load.host_f32(name, &[outputs, inputs])?,
            outputs,
            inputs,
        ))
    }

    /// A checkpoint vector as one f32 row `[1, len]`, such as a bias.
    pub fn row(&self, name: &str, len: usize) -> Result<Tensor<2>, WeightError> {
        Ok(self
            .load
            .upload_f32(self.load.host_f32(name, &[len])?, [1, len]))
    }
}
