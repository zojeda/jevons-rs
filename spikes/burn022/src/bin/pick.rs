#![forbid(unsafe_code)]
//! Item 5: custom CubeCL kernel as a Burn 0.22 backend extension (checked launch, no unsafe).
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::{Backend, Dispatch, backend_extension, tensor::{FloatTensor, IntTensor}};
use burn::tensor::{DType, Distribution, Int, Shape, Tensor, TensorData};
use burn022_spike::*;
use burn_cubecl::{CubeBackend, kernel::into_contiguous, ops::numeric::empty_device_dtype};
use cubecl::{CubeCount, CubeDim};

mod kern {
    use cubecl::prelude::*;

    /// out[p] = dot(hidden[pairs[p,0]], table[pairs[p,1]]) accumulated in f32.
    #[cube(launch)]
    pub fn pick_kernel<F: Float, I: Int>(
        hidden: &Tensor<F>,
        table: &Tensor<F>,
        pairs: &Tensor<I>,
        out: &mut Tensor<f32>,
        #[comptime] d: usize,
        #[define(F, I)] _dtypes: [ElemType; 2],
    ) {
        let p = CUBE_POS_X as usize;
        let row = usize::cast_from(pairs[2 * p]);
        let token = usize::cast_from(pairs[2 * p + 1]);
        let t = UNIT_POS_X as usize;
        let mut scratch = Shared::<[f32]>::new_slice(8usize);
        let mut acc = 0.0f32;
        #[unroll]
        for i in 0..comptime!(d / 256) {
            let c = i * 256 + t;
            acc += f32::cast_from(hidden[row * d + c]) * f32::cast_from(table[token * d + c]);
        }
        let s = plane_sum(acc);
        if UNIT_POS_X % 32 == 0 {
            scratch[(UNIT_POS_X / 32) as usize] = s;
        }
        sync_cube();
        if t == 0 {
            let mut total = 0.0f32;
            #[unroll]
            for w in 0..8usize {
                total += scratch[w];
            }
            out[p] = total;
        }
    }
}
use kern::pick_kernel;

#[backend_extension(Cube, Fusion)]
pub trait PickOps: Backend {
    #[fusion(dtype = { DType::F32 }, shape = { Shape::new([pairs[0]]) })]
    fn pick_logits(hidden: FloatTensor<Self>, table: FloatTensor<Self>, pairs: IntTensor<Self>) -> FloatTensor<Self>;
}

impl PickOps for CubeBackend {
    fn pick_logits(hidden: FloatTensor<Self>, table: FloatTensor<Self>, pairs: IntTensor<Self>) -> FloatTensor<Self> {
        let (hidden, table, pairs) = (into_contiguous(hidden), into_contiguous(table), into_contiguous(pairs));
        let d = hidden.meta.shape[1];
        assert_eq!(d % 256, 0);
        assert_eq!(table.meta.shape[1], d);
        let n = pairs.meta.shape[0];
        let out = empty_device_dtype(hidden.client.clone(), hidden.device.clone(), Shape::new([n]), DType::F32);
        let dtypes = [dtype_to_storage_type(hidden.dtype), dtype_to_storage_type(pairs.dtype)];
        let client = hidden.client.clone();
        pick_kernel::launch(
            &client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(256),
            hidden.into_tensor_arg(),
            table.into_tensor_arg(),
            pairs.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            d,
            dtypes,
        );
        out
    }
}

/// High-level wrapper over `Tensor<D>`.
pub fn pick_logits(hidden: Tensor<2>, table: Tensor<2>, pairs: Tensor<2, Int>) -> Tensor<1> {
    Tensor::from_dispatch(Dispatch::pick_logits(hidden.into_dispatch(), table.into_dispatch(), pairs.into_dispatch()))
}

fn main() {
    let device = rocm();
    let (rows, d, vocab, npairs) = (32usize, 4096usize, 65536usize, 2048usize);
    let hidden = Tensor::<2>::random([rows, d], Distribution::Uniform(-1.0, 1.0), (&device, DType::BF16));
    let table = Tensor::<2>::random([vocab, d], Distribution::Uniform(-0.05, 0.05), (&device, DType::BF16));
    let mut pv = Vec::with_capacity(npairs * 2);
    for p in 0..npairs {
        pv.push((p % rows) as i32);
        pv.push(((p * 7919) % vocab) as i32);
    }
    let pairs = Tensor::<2, Int>::from_data(TensorData::new(pv.clone(), [npairs, 2]), &device);
    let got = pick_logits(hidden.clone(), table.clone(), pairs.clone()).into_data().try_to_vec::<f32>().unwrap();
    // reference via Burn ops in f32
    let h32 = hidden.clone().cast(DType::F32).into_data().try_to_vec::<f32>().unwrap();
    let mut maxd = 0f32;
    for p in (0..npairs).step_by(97) {
        let (r, t) = (pv[2 * p] as usize, pv[2 * p + 1] as usize);
        let trow = table.clone().slice([t..t + 1, 0..d]).cast(DType::F32).into_data().try_to_vec::<f32>().unwrap();
        let refv: f32 = (0..d).map(|c| h32[r * d + c] * trow[c]).sum();
        maxd = maxd.max((refv - got[p]).abs());
    }
    println!("pick_logits custom kernel: {npairs} pairs, maxdiff vs CPU f32 ref (sampled) = {maxd:.3e}");
    // composes with fused ops before and after
    let fused = pick_logits(hidden.clone() * 1.0, table.clone(), pairs.clone()) * 2.0;
    let f = fused.into_data().try_to_vec::<f32>().unwrap();
    println!("fusion composition check: maxdiff(2*pick - 2*got) = {:.3e}", f.iter().zip(&got).map(|(a, b)| (a - 2.0 * b).abs()).fold(0f32, f32::max));
    let ms = bench(&device, 3, 20, || {
        let _ = pick_logits(hidden.clone(), table.clone(), pairs.clone());
    });
    println!("pick_logits {npairs} pairs d={d}: {ms:.3} ms ({:.1} GB/s of table rows)", (npairs * d * 2) as f64 / ms / 1e6);
    // compare with dense lm_head matmul for the same rows
    let tt = table.clone().transpose();
    let ms_dense = bench(&device, 2, 5, || {
        let _ = hidden.clone().matmul(tt.clone());
    });
    println!("dense [32x4096]x[4096x65536] for comparison: {ms_dense:.3} ms");
}
