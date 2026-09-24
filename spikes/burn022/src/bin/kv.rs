#![forbid(unsafe_code)]
//! Item 4: KV slab updated with slice_assign; in-place or copy? Plus raw bandwidth reference.
use burn::tensor::{DType, Distribution, Tensor};
use burn022_spike::*;

fn main() {
    let device = rocm();
    // bandwidth reference: 256 MiB elementwise read+write
    let mut big = Tensor::<1>::random([128 * 1024 * 1024], Distribution::Uniform(-1.0, 1.0), (&device, DType::BF16));
    let ms = bench_sync(&device, 3, 10, || {
        let b = std::mem::replace(&mut big, Tensor::<1>::empty([1], (&device, DType::BF16)));
        big = b * 0.5;
    });
    println!("elementwise x*2 on 256 MiB bf16: {ms:.3} ms -> {:.1} GB/s (read+write)", 512.0 * 1.048576 / ms);
    drop(big);
    device.memory_cleanup();

    for &cap in &[8192usize, 65536] {
        let mut slab = Tensor::<4>::zeros([1, 8, cap, 128], (&device, DType::BF16));
        let slab_mb = (8 * cap * 128 * 2) as f64 / 1048576.0;
        for &new in &[1usize, 32, 512] {
            let upd = Tensor::<4>::random([1, 8, new, 128], Distribution::Uniform(-1.0, 1.0), (&device, DType::BF16));
            let mut pos = 0usize;
            // sole owner: slab = slab.slice_assign(...)
            let ms = bench_sync(&device, 3, 20, || {
                let s = std::mem::replace(&mut slab, Tensor::<4>::empty([1, 1, 1, 1], (&device, DType::BF16)));
                slab = s.slice_assign([0..1, 0..8, pos..pos + new, 0..128], upd.clone());
                pos = (pos + new) % (cap - new);
            });
            // shared: another handle alive -> must copy
            let keep = slab.clone();
            let ms_shared = bench_sync(&device, 1, 5, || {
                let s = std::mem::replace(&mut slab, keep.clone());
                let _ = s.slice_assign([0..1, 0..8, 0..new, 0..128], upd.clone());
            });
            drop(keep);
            // full copy reference
            let mut other = slab.clone() + 0.0;
            let ms_copy = bench_sync(&device, 3, 10, || {
                let o = std::mem::replace(&mut other, Tensor::<4>::empty([1, 1, 1, 1], (&device, DType::BF16)));
                other = o * 0.5;
            });
            drop(other);
            println!(
                "kv slab cap={cap:6} ({slab_mb:6.1} MiB) new={new:4}: slice_assign sole-owner {ms:.4} ms | with extra handle alive {ms_shared:.4} ms | full slab copy {ms_copy:.4} ms"
            );
        }
        // correctness: write then read back a window
        let upd = Tensor::<4>::ones([1, 8, 4, 128], (&device, DType::BF16));
        slab = Tensor::<4>::zeros([1, 8, cap, 128], (&device, DType::BF16));
        slab = slab.slice_assign([0..1, 0..8, 100..104, 0..128], upd);
        let s: f32 = slab.clone().cast(DType::F32).sum().into_data().try_to_vec::<f32>().unwrap()[0];
        println!("  correctness: sum after writing 4 positions of ones = {s} (expect {})", 8 * 4 * 128);
        mem(&device, "  slab");
        drop(slab);
        device.memory_cleanup();
    }
}
