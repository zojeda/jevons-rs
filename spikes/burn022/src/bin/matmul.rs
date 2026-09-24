#![forbid(unsafe_code)]
//! Item 2: BF16 matmul accuracy, throughput, odd N, row invariance.
use burn::tensor::{DType, Tensor, TensorData};
use burn022_spike::*;
use half::bf16;

fn bf(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}
fn up(device: &burn::tensor::Device, v: &[f32], shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(TensorData::new(bf(v), shape), (device, DType::BF16))
}
fn read(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<bf16>().unwrap().iter().map(|x| x.to_f32()).collect()
}

fn main() {
    let device = rocm();
    // ---- accuracy
    for &(m, k, n) in &[(16usize, 256usize, 96usize), (33, 4096, 257)] {
        let a = rand_vec(m * k, 1);
        let b = rand_vec(k * n, 2);
        let ar: Vec<f32> = bf(&a).iter().map(|x| x.to_f32()).collect();
        let br: Vec<f32> = bf(&b).iter().map(|x| x.to_f32()).collect();
        let mut refc = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for p in 0..k {
                    s += ar[i * k + p] as f64 * br[p * n + j] as f64;
                }
                refc[i * n + j] = s as f32;
            }
        }
        let got = read(up(&device, &a, [m, k]).matmul(up(&device, &b, [k, n])));
        // also B given as [n,k] (weight layout) transposed
        let mut bt = vec![0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                bt[j * k + p] = b[p * n + j];
            }
        }
        let got_t = read(up(&device, &a, [m, k]).matmul(up(&device, &bt, [n, k]).transpose()));
        let (mut maxe, mut maxr, mut maxe_t) = (0f32, 0f32, 0f32);
        let rms = (refc.iter().map(|x| x * x).sum::<f32>() / refc.len() as f32).sqrt();
        for i in 0..refc.len() {
            maxe = maxe.max((got[i] - refc[i]).abs());
            maxe_t = maxe_t.max((got_t[i] - refc[i]).abs());
            maxr = maxr.max((got[i] - refc[i]).abs() / rms);
        }
        println!("accuracy m={m} k={k} n={n}: max_abs_err={maxe:.4e} (W^T view: {maxe_t:.4e}) ref_rms={rms:.3} max_err/rms={maxr:.3e} (bf16 ulp ~ 3.9e-3 rel)");
    }

    // ---- throughput
    let only_inv = std::env::var_os("ONLY_INV").is_some();
    let shapes: Vec<(usize, usize, usize, &str)> = if only_inv { vec![] } else { vec![
        (512, 4096, 14336, "mlp up"),
        (32, 4096, 131072, "lm_head even"),
        (32, 4096, 131073, "lm_head odd"),
        (1, 4096, 4096, "proj"),
        (8, 4096, 4096, "proj"),
        (32, 4096, 4096, "proj"),
        (128, 4096, 4096, "proj"),
        (512, 4096, 4096, "proj"),
        (2048, 4096, 4096, "proj"),
    ] };
    for (m, k, n, name) in shapes {
        let a = Tensor::<2>::random([m, k], burn::tensor::Distribution::Uniform(-1.0, 1.0), (&device, DType::BF16));
        // weight stored as [n, k] (PyTorch Linear layout), used transposed
        let w = Tensor::<2>::random([n, k], burn::tensor::Distribution::Uniform(-0.05, 0.05), (&device, DType::BF16));
        let wt = w.clone().transpose();
        let t0 = std::time::Instant::now();
        let _ = a.clone().matmul(wt.clone());
        device.sync().unwrap();
        let first = t0.elapsed().as_secs_f64() * 1e3;
        let ms = bench(&device, 3, 10, || {
            let _ = a.clone().matmul(wt.clone());
        });
        // contiguous [k, n] variant
        let wkn = Tensor::<2>::random([k, n], burn::tensor::Distribution::Uniform(-0.05, 0.05), (&device, DType::BF16));
        let ms2 = bench(&device, 3, 10, || {
            let _ = a.clone().matmul(wkn.clone());
        });
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let bytes = (n * k * 2) as f64;
        println!(
            "matmul {name:13} [{m}x{k}]x[{k}x{n}]: W^T view {ms:8.3} ms ({:6.2} TFLOP/s, {:6.1} GB/s weights) | contiguous KxN {ms2:8.3} ms | first call {first:.0} ms",
            flops / ms / 1e9,
            bytes / ms / 1e6
        );
        drop((a, w, wt, wkn));
        device.memory_cleanup();
    }

    // ---- row invariance
    let k = 4096;
    let n = 4096;
    let w = Tensor::<2>::random([n, k], burn::tensor::Distribution::Uniform(-0.05, 0.05), (&device, DType::BF16)).transpose();
    let base = rand_vec(2048 * k, 7);
    let ms = [1usize, 2, 3, 7, 16, 32, 33, 64, 128, 512, 2048];
    let outs: Vec<Vec<f32>> = ms
        .iter()
        .map(|&m| read(up(&device, &base[..m * k], [m, k]).matmul(w.clone()).slice([0..1, 0..n])))
        .collect();
    for (i, &m) in ms.iter().enumerate() {
        let row: Vec<String> = outs
            .iter()
            .map(|o| o.iter().zip(&outs[i]).filter(|(a, b)| a.to_bits() != b.to_bits()).count().to_string())
            .collect();
        println!("row-invariance diff counts of row0, M={m:5} vs M={ms:?}: {}", row.join(","));
    }
    // repeatability with same M
    let x = up(&device, &base[..32 * k], [32, k]);
    let a1 = read(x.clone().matmul(w.clone()));
    let a2 = read(x.matmul(w.clone()));
    println!("same-M repeat (M=32) bitwise identical: {}", a1.iter().zip(&a2).all(|(a, b)| a.to_bits() == b.to_bits()));
}

trait ContiguousLike {
    fn contiguous_like(self) -> Self;
}
impl ContiguousLike for Tensor<2> {
    fn contiguous_like(self) -> Self {
        // force a materialized copy in row-major layout
        let dims = self.dims();
        (self * 1.0).reshape(dims)
    }
}
