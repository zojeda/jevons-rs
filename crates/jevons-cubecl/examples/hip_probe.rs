//! Validated HIP upload, elementwise, reduction, and prefill-shaped float matmul.
use burn_rocm::{Rocm, RocmDevice};
use burn_tensor::{Tensor, TensorData, backend::Backend};
use serde_json::json;
use std::{error::Error, time::Instant};

type B = Rocm<f32>;

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 13 + salt) % 31) as f32 / 32.0 - 0.5)
        .collect()
}

fn sync(device: &RocmDevice) -> Result<(), Box<dyn Error>> {
    B::sync(device).map_err(|error| format!("HIP synchronization failed: {error:?}").into())
}

fn check_matmul(
    actual: &[f32],
    lhs: &[f32],
    rhs: &[f32],
    [m, k, n]: [usize; 3],
) -> Result<(usize, f64), Box<dyn Error>> {
    let mut checked = 0;
    let mut max_error = 0.0_f64;
    // Check all entries for the small case, 64 distributed entries for larger cases.
    for ri in 0..m.min(8) {
        let row = ri * (m - 1) / (m.min(8) - 1).max(1);
        for ci in 0..n.min(8) {
            let col = ci * (n - 1) / (n.min(8) - 1).max(1);
            let expected: f64 = (0..k)
                .map(|i| f64::from(lhs[row * k + i]) * f64::from(rhs[i * n + col]))
                .sum();
            let observed = f64::from(actual[row * n + col]);
            let error = (expected - observed).abs();
            if !observed.is_finite() || error > 1e-3 + expected.abs() * 1e-4 {
                return Err(
                    format!("Matmul mismatch at ({row}, {col}): {observed} != {expected}").into(),
                );
            }
            max_error = max_error.max(error);
            checked += 1;
        }
    }
    Ok((checked, max_error))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("Usage: hip_probe [ROUNDS=5] [DEVICE=0]".into());
    }
    let rounds = args.first().map_or(Ok(5), |arg| arg.parse::<usize>())?;
    if !(1..=1000).contains(&rounds) {
        return Err("ROUNDS must be in 1..=1000".into());
    }
    let device = RocmDevice::new(args.get(1).map_or(Ok(0), |arg| arg.parse::<usize>())?);
    let initialization = Instant::now();
    let input = values(257, 9);
    let tensor = Tensor::<B, 1>::from_data(TensorData::new(input.clone(), [257]), &device);
    let result = tensor * 2.0 + 1.0;
    let elementwise = result.clone().into_data().to_vec::<f32>()?;
    if !elementwise
        .iter()
        .zip(&input)
        .all(|(got, x)| *got == x * 2.0 + 1.0)
    {
        return Err("HIP elementwise/upload/readback validation failed".into());
    }
    let sum = result.sum().into_data().to_vec::<f32>()?[0];
    let expected_sum: f32 = input.iter().map(|x| x * 2.0 + 1.0).sum();
    if !sum.is_finite() || (sum - expected_sum).abs() > 1e-3 {
        return Err("HIP reduction validation failed".into());
    }
    let initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let mut cases = Vec::new();
    // Projection/expert dimensions from the inventoried DiffusionGemma GGUF.
    // These are float operator checks, not a simulation of routed expert traffic.
    for shape @ [m, k, n] in [
        [3, 7, 5],
        [128, 2816, 4096],
        [512, 2816, 4096],
        [128, 2816, 704],
    ] {
        let lhs = values(m * k, 3);
        let rhs = values(k * n, 11);
        let a = Tensor::<B, 2>::from_data(TensorData::new(lhs.clone(), [m, k]), &device);
        let b = Tensor::<B, 2>::from_data(TensorData::new(rhs.clone(), [k, n]), &device);
        sync(&device)?;
        let warmup = Instant::now();
        let output = a.clone().matmul(b.clone());
        sync(&device)?;
        let first_matmul_ms = warmup.elapsed().as_secs_f64() * 1000.0;
        check_matmul(&output.into_data().to_vec::<f32>()?, &lhs, &rhs, shape)?;
        let mut samples = Vec::new();
        for _ in 0..rounds {
            sync(&device)?;
            let start = Instant::now();
            let output = a.clone().matmul(b.clone());
            sync(&device)?;
            let matmul_ms = start.elapsed().as_secs_f64() * 1000.0;
            let (checked, error) =
                check_matmul(&output.into_data().to_vec::<f32>()?, &lhs, &rhs, shape)?;
            samples.push(
                json!({"matmul_ms": matmul_ms, "checked_entries": checked, "max_abs_error": error}),
            );
        }
        cases.push(
            json!({"shape_mkn": shape, "first_matmul_ms": first_matmul_ms, "samples": samples}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": "Burn 0.21.0 ROCm / CubeCL 0.10.0", "device_index": device.index,
            "dtype": "f32", "rounds": rounds, "initialization_ms": initialization_ms,
            "elementwise_and_reduction_passed": true, "cases": cases,
            "scope": "Float compute feasibility only; no GGUF, quantized matmul, or model inference. Synchronized host matmul time excludes upload and readback."
        }))?
    );
    Ok(())
}
