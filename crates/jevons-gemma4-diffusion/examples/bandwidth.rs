//! Raw device read bandwidth with coalesced and row-strided vector loads.
#![allow(clippy::unnecessary_cast)] // CubeCL DSL index casts
use cubecl::prelude::*;
use jevons_gemma4_diffusion::gpu::Gpu;
use std::time::Instant;

#[cube(launch)]
fn stream<N4: Size>(data: &[Vector<u32, N4>], out: &mut [u32], #[comptime] per_thread: usize) {
    let base = ABSOLUTE_POS as usize * per_thread;
    let mut acc = 0u32;
    #[unroll]
    for i in 0usize..per_thread {
        let v = data[base + i];
        acc = acc ^ v.extract(0usize) ^ v.extract(1usize) ^ v.extract(2usize) ^ v.extract(3usize);
    }
    out[ABSOLUTE_POS as usize] = acc;
}

/// Grid-stride coalesced: consecutive lanes read consecutive vectors.
#[cube(launch)]
fn stream_coalesced<N4: Size>(data: &[Vector<u32, N4>], out: &mut [u32], #[comptime] iters: usize) {
    let total = (CUBE_COUNT_X * CUBE_DIM_X) as usize;
    let mut acc = 0u32;
    #[unroll]
    for i in 0usize..iters {
        let v = data[i * total + ABSOLUTE_POS as usize];
        acc = acc ^ v.extract(0usize) ^ v.extract(1usize) ^ v.extract(2usize) ^ v.extract(3usize);
    }
    out[ABSOLUTE_POS as usize] = acc;
}

fn main() {
    let gpu = Gpu::new(0).unwrap();
    {
        let tiny = gpu.zeros(256, 4);
        let out = gpu.empty(1, 4);
        let run = || {
            stream::launch(
                &gpu.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                4,
                tiny.arg(),
                out.arg(),
                1usize,
            )
        };
        run();
        gpu.sync();
        for batch in [1usize, 10, 100, 1000] {
            let s = Instant::now();
            for _ in 0..batch {
                run();
            }
            gpu.sync();
            println!(
                "tiny kernel x{batch}: {:.1} us per launch",
                s.elapsed().as_secs_f64() * 1e6 / batch as f64
            );
        }
    }
    let words = 256usize << 20 >> 2; // 256 MB
    let data = gpu.upload_u32(&vec![1u32; words]);
    let vecs = words / 4;
    for (name, per) in [
        ("coalesced", 8usize),
        ("per-thread-contiguous 8", 8),
        ("per-thread-contiguous 2", 2),
    ] {
        let threads = vecs / per;
        let out = gpu.empty(threads, 4);
        let run = || {
            if name == "coalesced" {
                stream_coalesced::launch(
                    &gpu.client,
                    CubeCount::Static((threads / 256) as u32, 1, 1),
                    CubeDim::new_1d(256),
                    4,
                    data.arg(),
                    out.arg(),
                    per,
                );
            } else {
                stream::launch(
                    &gpu.client,
                    CubeCount::Static((threads / 256) as u32, 1, 1),
                    CubeDim::new_1d(256),
                    4,
                    data.arg(),
                    out.arg(),
                    per,
                );
            }
        };
        run();
        gpu.sync();
        let mut t = Vec::new();
        for _ in 0..10 {
            let s = Instant::now();
            for _ in 0..5 {
                run();
            }
            gpu.sync();
            t.push(s.elapsed().as_secs_f64() / 5.0);
        }
        t.sort_by(f64::total_cmp);
        println!("{name}: {:.0} GB/s", (words * 4) as f64 / t[5] / 1e9);
    }
}
