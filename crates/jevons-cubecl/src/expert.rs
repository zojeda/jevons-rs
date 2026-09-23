//! Experimental explicit FP16 tiles and fused packed Q4_K small-M products.
pub mod routed;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_tensor::{DType, FloatDType, Int, Tensor, TensorData, TensorPrimitive};
use cubecl::{hip::HipRuntime, prelude::*};
use cubek::{
    matmul::{
        components::tile::TileMatmulKind,
        definition::{
            MatmulElems, MatmulGlobalElems, MatmulProblem, TilingBlueprint, TilingScheme,
        },
        launch::Strategy,
        routines::BlueprintStrategy,
    },
    std::InputBinding,
};
use half::f16;

pub type DirectBackend = CubeBackend<HipRuntime, f32, i32, u8>;
type Matrix = Tensor<DirectBackend, 2>;

/// Tile configuration for the single-stage matrix instruction kernel.
#[derive(Clone, Copy, Debug)]
pub struct TileConfig {
    pub partition_m: u32,
    pub partition_n: u32,
    pub partition_k: u32,
    pub planes: u32,
}

/// Includes the same F32 input / F16 product / F32 output conversions as the control.
pub fn explicit_f16(input: Matrix, weights: Matrix, config: TileConfig) -> Result<Matrix, String> {
    let [m, k] = input.dims();
    let [n, weight_k] = weights.dims();
    if m == 0
        || n == 0
        || k == 0
        || k != weight_k
        || input.dtype() != DType::F32
        || weights.dtype() != DType::F16
        || input.device() != weights.device()
        || ![1, 2].contains(&config.partition_m)
        || ![1, 2, 4].contains(&config.partition_n)
        || ![2, 4].contains(&config.partition_k)
        || ![1, 2, 4].contains(&config.planes)
        || [m.checked_mul(k), n.checked_mul(k), m.checked_mul(n)]
            .iter()
            .any(|x| x.is_none_or(|v| v > u32::MAX as usize / 4))
    {
        return Err("Unsupported FP16 matrix shape, dtype, device or tile".into());
    }
    let lhs = input.cast(FloatDType::F16).into_primitive().tensor();
    let rhs = weights.transpose().into_primitive().tensor();
    let out = burn_cubecl::kernel::matmul::init_matmul_output(&lhs, &rhs, DType::F16);
    let globals = MatmulGlobalElems {
        lhs: DType::F16.into(),
        rhs: DType::F16.into(),
        out: DType::F16.into(),
    };
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs.meta.shape().clone(),
        rhs.meta.shape().clone(),
        out.meta.shape().clone(),
        lhs.meta.strides().clone(),
        rhs.meta.strides().clone(),
        out.meta.strides().clone(),
        globals.clone(),
        AddressType::U32,
        None,
        None,
    )
    .map_err(|e| format!("{e:?}"))?;
    let scheme = TilingScheme::builder()
        .with_tile_size((16, 16, 16).into())
        .with_partition_size((config.partition_m, config.partition_n, config.partition_k).into())
        .with_stage_size((config.planes, 1, 1).into())
        .build()
        .map_err(str::to_owned)?;
    let blueprint = TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, 32, &problem).build();
    cubek::matmul::launch::launch_ref(
        &Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(blueprint)),
        &lhs.client,
        InputBinding::new(lhs.clone().binding(), DType::F16.into()),
        InputBinding::new(rhs.binding(), DType::F16.into()),
        out.clone().binding(),
        &mut MatmulElems::from_globals(&globals),
    )
    .map_err(|e| format!("{e:?}"))?;
    Ok(Matrix::from_primitive(TensorPrimitive::Float(out)).cast(FloatDType::F32))
}

/// One wave handles one output feature and reuses unpacked weights across token rows.
/// Checked launch and explicit row guards keep partial tiles valid.
#[cube(launch)]
fn packed_product(
    quants: &Array<u8>,
    scales: &Array<f32>,
    minima: &Array<f32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] m: usize,
    #[comptime] k: usize,
    #[comptime] n: usize,
    #[comptime] tokens: usize,
) {
    let row = CUBE_POS_X as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    let first_token = CUBE_POS_Y as usize * tokens;
    let lane = UNIT_POS_X as usize;
    let mut acc = Array::<f32>::new(tokens);
    #[unroll]
    for t in 0..tokens {
        acc[t] = 0.0;
    }
    if row < n {
        for group in 0..k / 32 {
            let block = row * (k / 256) + group / 8;
            let sub = group % 8;
            let byte = u32::cast_from(quants[block * 128 + (sub / 2) * 32 + lane]);
            let q = (byte >> ((sub % 2) * 4) as u32) & 15;
            let w = f32::cast_from(q) * scales[block * 8 + sub] - minima[block * 8 + sub];
            #[unroll]
            for t in 0..tokens {
                if first_token + t < m {
                    acc[t] += w * input[(first_token + t) * k + group * 32 + lane];
                }
            }
        }
        #[unroll]
        for t in 0..tokens {
            let sum = plane_sum(acc[t]);
            if lane == 0 && first_token + t < m {
                output[(first_token + t) * n + row] = sum;
            }
        }
    }
}

/// Stage only a small Q4_K tile in shared memory, then use matrix instructions.
#[cube(launch)]
fn packed_cmma(
    quants: &Array<u8>,
    scales: &Array<f32>,
    minima: &Array<f32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] m: usize,
    #[comptime] k: usize,
    #[comptime] n: usize,
    #[comptime] waves: usize,
    #[comptime] stage_k: usize,
) {
    let first_row = CUBE_POS_X as usize * waves * 16;
    let first_token = CUBE_POS_Y as usize * 16;
    let wave = UNIT_POS_Y as usize;
    let unit = UNIT_POS_Y as usize * 32 + UNIT_POS_X as usize;
    let units = waves * 32;
    let mut left = SharedMemory::<f16>::new(16 * stage_k);
    let mut right = SharedMemory::<f16>::new(waves * 16 * stage_k);
    let mut scratch = SharedMemory::<f32>::new(waves * 256);
    let acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    for stage in 0..k / stage_k {
        for j in 0..(16 * stage_k).div_ceil(units) {
            let i = unit + j * units;
            if i < 16 * stage_k {
                let token = first_token + i / stage_k;
                let column = stage * stage_k + i % stage_k;
                let mut v = 0.0;
                if token < m {
                    v = input[token * k + column];
                }
                left[i] = f16::cast_from(v);
            }
        }
        for j in 0..(waves * 16 * stage_k) / units {
            let i = unit + j * units;
            let row = first_row + i / stage_k;
            let column = stage * stage_k + i % stage_k;
            let mut v = 0.0;
            if row < n {
                let block = row * (k / 256) + column / 256;
                let sub = (column % 256) / 32;
                let byte = u32::cast_from(quants[block * 128 + (sub / 2) * 32 + column % 32]);
                let q = (byte >> ((sub % 2) * 4) as u32) & 15;
                v = f32::cast_from(q) * scales[block * 8 + sub] - minima[block * 8 + sub];
            }
            right[i] = f16::cast_from(v);
        }
        sync_cube();
        #[unroll]
        for inner in 0..stage_k / 16 {
            let a = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::A,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
                &left.slice(inner * 16, 16 * stage_k),
                stage_k as u32,
            );
            let offset = wave * 16 * stage_k;
            let b = cmma::Matrix::<f16>::from_slice(
                cmma::MatrixIdent::B,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
                &right.slice(offset + inner * 16, offset + 16 * stage_k),
                stage_k as u32,
            );
            cmma::execute(&a, &b, &acc, &acc);
        }
        sync_cube();
    }
    cmma::store(
        &mut scratch.slice_mut(wave * 256, (wave + 1) * 256),
        &acc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    for j in 0..waves * 256 / units {
        let i = unit + j * units;
        let token = first_token + (i % 256) / 16;
        let row = first_row + (i / 256) * 16 + i % 16;
        if token < m && row < n {
            output[token * n + row] = scratch[i];
        }
    }
}

/// Packed nibbles plus expanded scale/min metadata (192 bytes per 256 weights).
pub struct PackedExpert {
    quants: CubeTensor<HipRuntime>,
    scales: CubeTensor<HipRuntime>,
    minima: CubeTensor<HipRuntime>,
    n: usize,
    k: usize,
}

impl PackedExpert {
    pub fn upload(
        bytes: &[u8],
        n: usize,
        k: usize,
        device: &burn_rocm::RocmDevice,
    ) -> Result<Self, String> {
        if n == 0
            || k == 0
            || !k.is_multiple_of(256)
            || n.checked_mul(k).is_none_or(|v| v > u32::MAX as usize / 4)
            || n.checked_mul(k / 256).and_then(|v| v.checked_mul(144)) != Some(bytes.len())
        {
            return Err("Invalid Q4_K expert shape or byte count".into());
        }
        let mut q = Vec::new();
        let mut scales = Vec::new();
        let mut minima = Vec::new();
        for block in bytes.as_chunks::<144>().0 {
            let (s, m) = crate::q4k::block_scales(block);
            if s.iter().chain(&m).any(|x| !x.is_finite()) {
                return Err("Nonfinite Q4_K metadata".into());
            }
            q.extend_from_slice(&block[16..]);
            scales.extend(s);
            minima.extend(m);
        }
        let len = q.len();
        let quants = Tensor::<DirectBackend, 1, Int>::from_data(
            TensorData::new(q, [len]),
            (device, DType::U8),
        )
        .into_primitive();
        let upload = |data: Vec<f32>| {
            let len = data.len();
            Tensor::<DirectBackend, 1>::from_data(TensorData::new(data, [len]), device)
                .into_primitive()
                .tensor()
        };
        Ok(Self {
            quants,
            scales: upload(scales),
            minima: upload(minima),
            n,
            k,
        })
    }

    pub fn matmul(&self, input: Matrix, tokens: usize, waves: u32) -> Result<Matrix, String> {
        let [m, k] = input.dims();
        if m == 0
            || k != self.k
            || input.dtype() != DType::F32
            || input.device() != self.quants.device
            || m.checked_mul(k).is_none_or(|v| v > u32::MAX as usize / 4)
            || m.checked_mul(self.n)
                .is_none_or(|v| v > u32::MAX as usize / 4)
            || ![1, 2, 4, 8].contains(&tokens)
            || ![1, 2, 4, 8].contains(&waves)
        {
            return Err("Unsupported input or fused tile configuration".into());
        }
        let input = input.into_primitive().tensor();
        if input.client.properties().hardware.plane_size_min != 32
            || input.client.properties().hardware.plane_size_max != 32
        {
            return Err("This fused prototype requires 32-lane waves".into());
        }
        if input.meta.strides().as_ref() != [k, 1] {
            return Err("Input must be contiguous".into());
        }
        // Allocate a flat buffer: a 2D allocation may pad row strides (e.g. N=5).
        let out = Tensor::<DirectBackend, 1>::empty([m * self.n], &input.device)
            .reshape([m, self.n])
            .into_primitive()
            .tensor();
        packed_product::launch::<HipRuntime>(
            &input.client,
            CubeCount::Static(
                (self.n as u32).div_ceil(waves),
                m.div_ceil(tokens) as u32,
                1,
            ),
            CubeDim::new_2d(32, waves),
            self.quants.clone().into_array_arg(),
            self.scales.clone().into_array_arg(),
            self.minima.clone().into_array_arg(),
            input.clone().into_array_arg(),
            out.clone().into_array_arg(),
            m,
            k,
            self.n,
            tokens,
        );
        Ok(Matrix::from_primitive(TensorPrimitive::Float(out)))
    }

    pub fn matmul_cmma(&self, input: Matrix, waves: u32, stage_k: usize) -> Result<Matrix, String> {
        let [m, k] = input.dims();
        if m == 0
            || k != self.k
            || input.dtype() != DType::F32
            || input.device() != self.quants.device
            || m.checked_mul(k).is_none_or(|v| v > u32::MAX as usize / 4)
            || m.checked_mul(self.n)
                .is_none_or(|v| v > u32::MAX as usize / 4)
            || ![2, 4, 8].contains(&waves)
            || ![32, 64].contains(&stage_k)
        {
            return Err("Unsupported CMMA input or configuration".into());
        }
        let input = input.into_primitive().tensor();
        if input.meta.strides().as_ref() != [k, 1]
            || input.client.properties().hardware.plane_size_min != 32
            || input.client.properties().hardware.plane_size_max != 32
        {
            return Err("CMMA prototype requires contiguous input and 32-lane waves".into());
        }
        if !TileMatmulKind::Cmma.is_supported(
            &input.client,
            cubecl::features::MmaConfig {
                a_type: DType::F16.into(),
                b_type: DType::F16.into(),
                cd_type: DType::F32.into(),
                m: 16,
                n: 16,
                k: 16,
            },
        ) {
            return Err("Device lacks F16/F32 16x16x16 matrix instructions".into());
        }
        let out = Tensor::<DirectBackend, 1>::empty([m * self.n], &input.device)
            .reshape([m, self.n])
            .into_primitive()
            .tensor();
        packed_cmma::launch::<HipRuntime>(
            &input.client,
            CubeCount::Static(
                (self.n as u32).div_ceil(waves * 16),
                m.div_ceil(16) as u32,
                1,
            ),
            CubeDim::new_2d(32, waves),
            self.quants.clone().into_array_arg(),
            self.scales.clone().into_array_arg(),
            self.minima.clone().into_array_arg(),
            input.clone().into_array_arg(),
            out.clone().into_array_arg(),
            m,
            k,
            self.n,
            waves as usize,
            stage_k,
        );
        Ok(Matrix::from_primitive(TensorPrimitive::Float(out)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_packed_expert_shapes_fail_before_gpu_access() {
        let device = burn_rocm::RocmDevice::new(0);
        for (n, k, bytes) in [
            (0, 256, vec![]),
            (1, 255, vec![0; 144]),
            (1, 256, vec![0; 143]),
            (usize::MAX, 256, vec![]),
        ] {
            assert!(PackedExpert::upload(&bytes, n, k, &device).is_err());
        }
    }

    #[test]
    #[ignore = "requires a HIP device with 32-lane waves"]
    fn fused_tiles_preserve_partial_token_and_output_rows() {
        let device = burn_rocm::RocmDevice::new(0);
        let (m, k, n) = (3, 512, 5);
        let mut raw = vec![0u8; n * k / 256 * 144];
        for (b, block) in raw.as_chunks_mut::<144>().0.iter_mut().enumerate() {
            block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
            block[4..16].fill(1);
            for (i, q) in block[16..].iter_mut().enumerate() {
                *q = (i * 17 + b * 13) as u8;
            }
        }
        let x: Vec<f32> = (0..m * k)
            .map(|i| (i as i32 % 17 - 8) as f32 * 0.05)
            .collect();
        let mut reference = vec![0.0_f64; m * n];
        for t in 0..m {
            for row in 0..n {
                for column in 0..k {
                    let block = row * (k / 256) + column / 256;
                    let group = (column % 256) / 32;
                    let byte = raw[block * 144 + 16 + (group / 2) * 32 + column % 32];
                    let quant = (byte >> ((group % 2) * 4)) & 15;
                    reference[t * n + row] += f64::from(x[t * k + column]) * f64::from(quant) * 0.5;
                }
            }
        }
        let packed = PackedExpert::upload(&raw, n, k, &device).unwrap();
        let input = Matrix::from_data(TensorData::new(x, [m, k]), &device);
        for tokens in [1, 2, 4, 8] {
            for waves in [4, 8] {
                let actual = packed
                    .matmul(input.clone(), tokens, waves)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for (a, b) in actual.iter().zip(&reference) {
                    assert!(
                        (f64::from(*a) - b).abs() < 1e-4,
                        "{tokens}/{waves}: {a} != {b}"
                    );
                }
            }
        }
        assert!(packed.matmul(input, 3, 4).is_err());
    }

    #[test]
    #[ignore = "requires HIP 16x16x16 matrix instructions"]
    fn fused_cmma_preserves_partial_tiles() {
        let device = burn_rocm::RocmDevice::new(0);
        let (m, k, n) = (3, 256, 5);
        let mut raw = vec![0u8; n * k / 256 * 144];
        for (row, block) in raw.as_chunks_mut::<144>().0.iter_mut().enumerate() {
            block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
            block[4..16].fill(1);
            block[16..].fill((((row + 2) << 4) | (row + 1)) as u8);
        }
        let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.25).collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|i| {
                let (t, row) = (i / n, i % n);
                (0..k)
                    .map(|col| {
                        x[t * k + col]
                            * if (col / 32) % 2 == 0 {
                                (row + 1) as f32 * 0.5
                            } else {
                                (row + 2) as f32 * 0.5
                            }
                    })
                    .sum()
            })
            .collect();
        let input = Matrix::from_data(TensorData::new(x, [m, k]), &device);
        let packed = PackedExpert::upload(&raw, n, k, &device).unwrap();
        for waves in [2, 4, 8] {
            for stage_k in [32, 64] {
                let actual = packed
                    .matmul_cmma(input.clone(), waves, stage_k)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for (i, a) in actual.iter().enumerate() {
                    assert!(
                        (*a - expected[i]).abs() < 1e-4,
                        "{waves}/{stage_k}: {a} != {}",
                        expected[i]
                    );
                }
            }
        }
    }
}
