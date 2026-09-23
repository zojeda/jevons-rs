//! GPU grouping and fused expert products. IDs are already selected by the router.
use super::{DirectBackend, Matrix, PackedExpert};
use burn_cubecl::tensor::CubeTensor;
use burn_tensor::{DType, Int, Tensor, TensorData, TensorPrimitive};
use cubecl::{hip::HipRuntime, prelude::*};

#[cube(launch)]
fn group_ids(
    ids: &Array<u32>,
    counts: &mut Array<Atomic<u32>>,
    slots: &mut Array<u32>,
    #[comptime] assignments: usize,
    #[comptime] m: usize,
) {
    let i = ABSOLUTE_POS;
    if i < assignments {
        let expert = ids[i] as usize;
        let position = counts[expert].fetch_add(1) as usize;
        slots[expert * m + position] = i as u32;
    }
}

#[cube(launch)]
fn make_jobs(
    counts: &Array<u32>,
    jobs: &mut Array<u32>,
    job_count: &mut Array<u32>,
    #[comptime] experts: usize,
    #[comptime] tokens: usize,
) {
    let expert = UNIT_POS as usize;
    if expert < experts {
        let mut start = 0u32;
        for e in 0..expert {
            start += counts[e].div_ceil(tokens as u32);
        }
        let count = counts[expert].div_ceil(tokens as u32);
        for j in 0..count as usize {
            jobs[(start as usize + j) * 2] = expert as u32;
            jobs[(start as usize + j) * 2 + 1] = (j * tokens) as u32;
        }
        if expert == experts - 1 {
            job_count[0] = start + count;
        }
    }
}

#[cube(launch)]
fn grouped_product(
    quants: &Array<u8>,
    scales: &Array<f32>,
    minima: &Array<f32>,
    input: &Array<f32>,
    counts: &Array<u32>,
    slots: &Array<u32>,
    jobs: &Array<u32>,
    job_count: &Array<u32>,
    output: &mut Array<f32>,
    #[comptime] m: usize,
    #[comptime] k: usize,
    #[comptime] n: usize,
    #[comptime] top_k: usize,
    #[comptime] tokens: usize,
) {
    let job = CUBE_POS_Y as usize;
    if job >= job_count[0] as usize {
        terminate!();
    }
    let expert = jobs[job * 2] as usize;
    let first = jobs[job * 2 + 1] as usize;
    let row = CUBE_POS_X as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let mut acc = Array::<f32>::new(tokens);
    let mut assignments = Array::<u32>::new(tokens);
    let mut input_offsets = Array::<u32>::new(tokens);
    let count = counts[expert] as usize;
    // Resolve routing once per tile rather than inside every reduction step.
    // Inactive tail entries read the valid first input row; their sums are never
    // scattered. This keeps the reduction free of per-token routing branches.
    #[unroll]
    for t in 0..tokens {
        acc[t] = 0.0;
        assignments[t] = 0;
        input_offsets[t] = 0;
        if first + t < count {
            let assignment = slots[expert * m + first + t];
            assignments[t] = assignment;
            input_offsets[t] = (assignment / top_k as u32) * k as u32;
        }
    }
    if row < n {
        // Unroll the eight subgroups in a Q4_K block, exposing independent
        // loads and constant nibble offsets while keeping the K loop bounded.
        for block_k in 0..k / 256 {
            #[unroll]
            for sub in 0usize..8 {
                let group = block_k * 8 + sub;
                let block = (expert * n + row) * (k / 256) + block_k;
                let byte = u32::cast_from(quants[block * 128 + (sub / 2) * 32 + lane]);
                let q = (byte >> ((sub % 2) * 4) as u32) & 15;
                let w = f32::cast_from(q) * scales[block * 8 + sub] - minima[block * 8 + sub];
                #[unroll]
                for t in 0..tokens {
                    acc[t] += w * input[input_offsets[t] as usize + group * 32 + lane];
                }
            }
        }
        #[unroll]
        for t in 0..tokens {
            let sum = plane_sum(acc[t]);
            if lane == 0 && first + t < count {
                let assignment = assignments[t] as usize;
                output[assignment * n + row] = sum;
            }
        }
    }
}

/// Validated top-k IDs. Upload is outside the timed resident-operator benchmark.
pub struct Routes {
    ids: CubeTensor<HipRuntime>,
    m: usize,
    experts: usize,
    top_k: usize,
}
impl Routes {
    pub fn upload(
        ids: Vec<u32>,
        m: usize,
        experts: usize,
        top_k: usize,
        device: &burn_rocm::RocmDevice,
    ) -> Result<Self, String> {
        if m == 0
            || !(1..=128).contains(&experts)
            || top_k == 0
            || top_k > experts
            || m.checked_mul(top_k) != Some(ids.len())
            || m.checked_mul(experts)
                .is_none_or(|v| v > u32::MAX as usize / 4)
            || ids.len() > u32::MAX as usize / 4
        {
            return Err("Invalid route dimensions".into());
        }
        for token in ids.chunks_exact(top_k) {
            for (i, &id) in token.iter().enumerate() {
                if id as usize >= experts || token[..i].contains(&id) {
                    return Err("Expert IDs must be in range and distinct per token".into());
                }
            }
        }
        let len = ids.len();
        let ids = Tensor::<DirectBackend, 1, Int>::from_data(
            TensorData::new(ids, [len]),
            (device, DType::U32),
        )
        .into_primitive();
        Ok(Self {
            ids,
            m,
            experts,
            top_k,
        })
    }
}

impl PackedExpert {
    /// Includes count reset, GPU grouping, tile-job construction, matmul, and output scatter.
    pub fn matmul_routed(
        &self,
        input: Matrix,
        routes: &Routes,
        tokens: usize,
        waves: u32,
    ) -> Result<Tensor<DirectBackend, 3>, String> {
        let [m, k] = input.dims();
        let n = self.n / routes.experts;
        if m != routes.m
            || k != self.k
            || !self.n.is_multiple_of(routes.experts)
            || input.dtype() != DType::F32
            || input.device() != self.quants.device
            || input.device() != routes.ids.device
            || ![4, 8].contains(&tokens)
            || ![4, 8].contains(&waves)
            || m.checked_mul(k).is_none_or(|v| v > u32::MAX as usize / 4)
            || m.checked_mul(routes.top_k)
                .and_then(|v| v.checked_mul(n))
                .is_none_or(|v| v > u32::MAX as usize / 4)
        {
            return Err("Unsupported routed shape, device or configuration".into());
        }
        let input = input.into_primitive().tensor();
        if input.meta.strides().as_ref() != [k, 1]
            || input.client.properties().hardware.plane_size_min != 32
            || input.client.properties().hardware.plane_size_max != 32
        {
            return Err("Routed prototype requires contiguous input and 32-lane waves".into());
        }
        let assignments = m * routes.top_k;
        // sum(ceil(count[e]/tokens)) <= ceil(assignments/tokens) + experts.
        // Actual counts drive jobs; no expert is capped at the average workload.
        let max_jobs = assignments.div_ceil(tokens) + routes.experts;
        if max_jobs > input.client.properties().hardware.max_cube_count.1 as usize {
            return Err("Routed job count exceeds this device's dispatch limit".into());
        }
        let counts =
            Tensor::<DirectBackend, 1, Int>::zeros([routes.experts], (&input.device, DType::U32))
                .into_primitive();
        let empty = |len| {
            Tensor::<DirectBackend, 1, Int>::empty([len], (&input.device, DType::U32))
                .into_primitive()
        };
        let slots = empty(routes.experts * m);
        let jobs = empty(max_jobs * 2);
        let job_count = empty(1);
        let out = Tensor::<DirectBackend, 1>::empty([assignments * n], &input.device)
            .reshape([m, routes.top_k, n])
            .into_primitive()
            .tensor();
        group_ids::launch::<HipRuntime>(
            &input.client,
            CubeCount::Static(assignments.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            routes.ids.clone().into_array_arg(),
            counts.clone().into_array_arg(),
            slots.clone().into_array_arg(),
            assignments,
            m,
        );
        make_jobs::launch::<HipRuntime>(
            &input.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(128),
            counts.clone().into_array_arg(),
            jobs.clone().into_array_arg(),
            job_count.clone().into_array_arg(),
            routes.experts,
            tokens,
        );
        grouped_product::launch::<HipRuntime>(
            &input.client,
            CubeCount::Static((n as u32).div_ceil(waves), max_jobs as u32, 1),
            CubeDim::new_2d(32, waves),
            self.quants.clone().into_array_arg(),
            self.scales.clone().into_array_arg(),
            self.minima.clone().into_array_arg(),
            input.clone().into_array_arg(),
            counts.into_array_arg(),
            slots.into_array_arg(),
            jobs.into_array_arg(),
            job_count.into_array_arg(),
            out.clone().into_array_arg(),
            m,
            k,
            n,
            routes.top_k,
            tokens,
        );
        Ok(Tensor::from_primitive(TensorPrimitive::Float(out)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_routes_fail_before_gpu_access() {
        let d = burn_rocm::RocmDevice::new(0);
        for ids in [vec![0, 0], vec![0, 5], vec![0]] {
            assert!(Routes::upload(ids, 1, 5, 2, &d).is_err());
        }
    }
    #[test]
    #[ignore = "requires a 32-lane HIP GPU"]
    fn skewed_routes_preserve_all_assignments_and_empty_experts() {
        let d = burn_rocm::RocmDevice::new(0);
        let (m, k, n, experts, top_k) = (5, 256, 5, 5, 2);
        let ids = vec![0, 1, 0, 1, 2, 0, 3, 0, 0, 2];
        let mut raw = vec![0u8; experts * n * k / 256 * 144];
        for (row, b) in raw.as_chunks_mut::<144>().0.iter_mut().enumerate() {
            b[..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
            b[4..16].fill(1);
            let q = (row % 15 + 1) as u8;
            b[16..].fill(q | (q << 4));
        }
        let x: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 * 0.25).collect();
        let sums: Vec<f32> = x.chunks_exact(k).map(|row| row.iter().sum()).collect();
        let expected: Vec<f32> = ids
            .iter()
            .enumerate()
            .flat_map(|(a, &e)| {
                let sums = &sums;
                (0..n).map(move |row| {
                    sums[a / top_k] * ((e as usize * n + row) % 15 + 1) as f32 * 0.25
                })
            })
            .collect();
        let routes = Routes::upload(ids, m, experts, top_k, &d).unwrap();
        let packed = PackedExpert::upload(&raw, experts * n, k, &d).unwrap();
        let input = Matrix::from_data(TensorData::new(x, [m, k]), &d);
        for tokens in [4, 8] {
            for waves in [4, 8] {
                for _ in 0..3 {
                    let actual = packed
                        .matmul_routed(input.clone(), &routes, tokens, waves)
                        .unwrap()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
