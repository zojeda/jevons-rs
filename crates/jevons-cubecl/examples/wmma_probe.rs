//! Matrix-instruction throughput and workgroup concurrency probe.
#![allow(clippy::unnecessary_cast)] // CubeCL DSL index casts
use cubecl::prelude::*;
use half::f16;
use jevons_cubecl::gpu::{Gpu, Hip};
use std::time::Instant;

#[cube(launch)]
fn wmma_loop<N8: Size>(out: &mut Array<f32>, iters: u32, #[comptime] use_lds: bool) {
    let def = cmma::MmaDefinition::<f16, f16, f32>::new(16usize, 16usize, 16usize);
    let size!(NC) = def.vector_size(cmma::MatrixIdent::Accumulator);
    let mut a = Array::<Vector<f16, N8>>::new(2usize);
    let mut b = Array::<Vector<f16, N8>>::new(2usize);
    let v = Vector::<f16, N8>::empty().fill(f16::cast_from(f32::cast_from(UNIT_POS) * 1.0e-3f32));
    a[0usize] = v;
    a[1usize] = v;
    b[0usize] = v;
    b[1usize] = v;
    let mut lds = SharedMemory::<Vector<f16, N8>>::new(256usize);
    lds[UNIT_POS as usize % 256usize] = v;
    sync_cube();
    let mut acc = Sequence::<Array<Vector<f32, NC>>>::new();
    #[unroll]
    for _i in 0usize..4usize {
        let mut c = Array::<Vector<f32, NC>>::new(8usize);
        #[unroll]
        for e in 0usize..8usize {
            c[e] = Vector::cast_from(0.0f32);
        }
        acc.push(c);
    }
    for it in 0..iters {
        if comptime!(use_lds) {
            let idx = (it as usize * 2usize + UNIT_POS_PLANE as usize) % 255usize;
            b[0usize] = lds[idx];
            b[1usize] = lds[idx + 1usize];
        }
        #[unroll]
        for i in 0usize..4usize {
            def.execute_inplace(&a, &b, acc.index_mut(i));
        }
    }
    let mut s = 0.0f32;
    #[unroll]
    for i in 0usize..4usize {
        s += acc.index(i)[0usize][0usize];
    }
    out[ABSOLUTE_POS as usize] = s;
}

#[cube(launch)]
fn fma_loop(out: &mut Array<f32>, iters: u32) {
    let mut a = Array::<f32>::new(8usize);
    #[unroll]
    for i in 0usize..8usize {
        a[i] = f32::cast_from(UNIT_POS + i as u32);
    }
    let m = f32::cast_from(ABSOLUTE_POS) * 1.0e-9f32 + 0.999f32;
    for _ in 0..iters {
        #[unroll]
        for i in 0usize..8usize {
            a[i] = a[i] * m + 0.5f32;
        }
    }
    let mut s = 0.0f32;
    #[unroll]
    for i in 0usize..8usize {
        s += a[i];
    }
    out[ABSOLUTE_POS as usize] = s;
}

fn main() {
    let gpu = Gpu::new(0).unwrap();
    {
        let groups = 4000u32;
        let iters = 4096u32;
        let out = gpu.empty(groups as usize * 256, 4);
        let run = || {
            fma_loop::launch::<Hip>(
                &gpu.client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(256),
                out.arg(),
                iters,
            )
        };
        run();
        gpu.sync();
        let s = Instant::now();
        for _ in 0..5 {
            run();
        }
        gpu.sync();
        let t = s.elapsed().as_secs_f64() / 5.0;
        let fmas = groups as f64 * 256.0 * iters as f64 * 8.0;
        println!(
            "f32 FMA: {:.2} TFLOP/s -> {:.2} GHz equivalent at 2560 lanes x 1 FMA/clk",
            fmas * 2.0 / t / 1e12,
            fmas / t / 2560.0 / 1e9
        );
    }
    let iters = 4096u32;
    for use_lds in [false, true] {
        for groups in [1u32, 40, 80, 160, 400, 1600] {
            let out = gpu.empty(groups as usize * 128, 4);
            let run = || {
                wmma_loop::launch::<Hip>(
                    &gpu.client,
                    CubeCount::Static(groups, 1, 1),
                    CubeDim::new_2d(32, 4),
                    8,
                    out.arg(),
                    iters,
                    use_lds,
                )
            };
            run();
            gpu.sync();
            let s = Instant::now();
            for _ in 0..5 {
                run();
            }
            gpu.sync();
            let t = s.elapsed().as_secs_f64() / 5.0;
            let wmmas = groups as f64 * 4.0 * iters as f64 * 4.0;
            println!(
                "lds={use_lds} groups={groups}: {:.3} ms, {:.1} TFLOP/s, {:.1} cycles/wmma/wave at 2.9 GHz",
                t * 1e3,
                wmmas * 8192.0 / t / 1e12,
                t * 2.9e9 / (iters as f64 * 4.0)
            );
        }
    }
}
