//! Codegen probe: does CubeCL HIP output compile to RDNA3 int8 dot-product instructions?
#![allow(clippy::unnecessary_cast)] // CubeCL DSL index casts
use cubecl::prelude::*;
use jevons_gemma4_diffusion::gpu::Gpu;

/// Sign-extended byte products written with shifts on i32.
#[cube(launch)]
fn dot_shifts(a: &[u32], b: &[u32], out: &mut [i32]) {
    let i = ABSOLUTE_POS as usize;
    let (x, y) = (i32::cast_from(a[i]), i32::cast_from(b[i]));
    let mut acc = 0i32;
    #[unroll]
    for k in 0u32..4u32 {
        let sh = comptime!((24 - 8 * k) as i32);
        acc += ((x << sh) >> 24i32) * ((y << sh) >> 24i32);
    }
    out[i] = acc;
}

/// The same through vectorized i8 and CubeCL's dot.
#[cube(launch)]
fn dot_vector<N4: Size>(a: &[u32], b: &[u32], out: &mut [i32]) {
    let i = ABSOLUTE_POS as usize;
    let x = Vector::<i8, N4>::reinterpret(a[i]);
    let y = Vector::<i8, N4>::reinterpret(b[i]);
    let xi = Vector::<i32, N4>::cast_from(x);
    let yi = Vector::<i32, N4>::cast_from(y);
    out[i] = xi.dot(yi);
}

fn main() {
    let gpu = Gpu::new(0).unwrap();
    let n = 1024usize;
    let a: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();
    let b: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(40503).rotate_left(7))
        .collect();
    let want: Vec<i32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            (0..4)
                .map(|k| {
                    i32::from((x >> (8 * k)) as u8 as i8) * i32::from((y >> (8 * k)) as u8 as i8)
                })
                .sum()
        })
        .collect();
    let (ab, bb) = (gpu.upload_u32(&a), gpu.upload_u32(&b));
    for variant in ["shifts", "vector"] {
        let out = gpu.zeros(n, 4);
        let count = CubeCount::Static((n / 256) as u32, 1, 1);
        if variant == "shifts" {
            dot_shifts::launch(
                &gpu.client,
                count,
                CubeDim::new_1d(256),
                ab.arg(),
                bb.arg(),
                out.arg(),
            );
        } else {
            dot_vector::launch(
                &gpu.client,
                count,
                CubeDim::new_1d(256),
                4,
                ab.arg(),
                bb.arg(),
                out.arg(),
            );
        }
        let got: Vec<i32> = gpu.read_u32(&out).into_iter().map(|v| v as i32).collect();
        println!(
            "{variant}: mismatches {}",
            got.iter().zip(&want).filter(|(g, w)| g != w).count()
        );
    }
}
