#![forbid(unsafe_code)]
//! Item 1: HIP init, naming, cache location, memory release on drop.
use burn::tensor::{DType, Tensor, TensorData};
use burn022_spike::*;
use std::time::Instant;

fn main() {
    let t = Instant::now();
    let device = rocm();
    println!("device = {device:?}");
    let x = Tensor::<2>::zeros([4, 4], (&device, DType::BF16));
    device.sync().unwrap();
    println!("first tensor + sync: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    println!("identity = {:?}", device.identity());
    println!("settings = {:?}", device.settings());
    println!("x dtype = {:?}", x.dtype());
    mem(&device, "after tiny");

    // 1 GiB bf16 tensors
    let n = 512 * 1024 * 1024; // elements -> 1 GiB bf16
    {
        let a = Tensor::<1>::ones([n], (&device, DType::BF16));
        let b = a.clone() * 2.0;
        device.sync().unwrap();
        mem(&device, "with 2x1GiB live");
        drop(a);
        drop(b);
        device.sync().unwrap();
        mem(&device, "after drop (no cleanup)");
        device.memory_cleanup();
        device.sync().unwrap();
        mem(&device, "after memory_cleanup");
    }
    // Persistent allocations (for model weights)
    let w = device.memory_persistent_allocations((), |_| {
        Tensor::<2>::from_data(
            TensorData::new(vec![half::bf16::from_f32(1.0); 4096 * 4096], [4096, 4096]),
            (&device, DType::BF16),
        )
    });
    device.sync().unwrap();
    mem(&device, "persistent 32MiB weight live");
    drop(w);
    device.memory_cleanup();
    device.sync().unwrap();
    mem(&device, "persistent dropped + cleanup");
    let t = Instant::now();
    let y = (x.clone() + 1.0).exp().sum();
    println!("small fused op result = {:?} ({:.1} ms incl compile)", y.into_data().try_to_vec::<half::bf16>().unwrap(), t.elapsed().as_secs_f64() * 1e3);
}
