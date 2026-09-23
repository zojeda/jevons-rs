//! Compare real GGUF Q4_K matrix products against exported CPU reference values.
use burn_rocm::RocmDevice;
use burn_tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use jevons_cubecl::q4k::{HipBackend as B, Q4kWeight};
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, time::Instant};

fn f32_file(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("Invalid f32 file length".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn sync(device: &RocmDevice) -> Result<(), Box<dyn Error>> {
    B::sync(device).map_err(|e| format!("HIP synchronization failed: {e:?}").into())
}

fn errors(actual: &[f32], reference: &[f32]) -> Result<Value, Box<dyn Error>> {
    if actual.len() != reference.len() || actual.is_empty() {
        return Err("Output shape mismatch".into());
    }
    let (mut squared, mut norm, mut max_error, mut max_reference) = (0.0, 0.0, 0.0_f64, 0.0_f64);
    for (&a, &b) in actual.iter().zip(reference) {
        if !a.is_finite() || !b.is_finite() {
            return Err("Nonfinite output".into());
        }
        let diff = f64::from(a) - f64::from(b);
        squared += diff * diff;
        norm += f64::from(b).powi(2);
        max_error = max_error.max(diff.abs());
        max_reference = max_reference.max(f64::from(b).abs());
    }
    Ok(json!({"relative_rmse": (squared / norm.max(1e-30)).sqrt(),
              "max_abs_error": max_error, "normalized_max_error": max_error / max_reference.max(1e-30)}))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("Usage: q4k_bench FIXTURE_DIRECTORY ROUNDS".into());
    }
    let dir = Path::new(&args[0]);
    let rounds: usize = args[1].parse()?;
    if !(1..=1000).contains(&rounds) {
        return Err("Rounds must be in 1..=1000".into());
    }
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    let device = RocmDevice::new(0);
    let mut cases = Vec::new();
    for case in manifest["cases"].as_array().ok_or("Missing cases")? {
        let m = case["m"].as_u64().ok_or("Missing m")? as usize;
        let k = case["k"].as_u64().ok_or("Missing k")? as usize;
        let n = case["n"].as_u64().ok_or("Missing n")? as usize;
        let file = |key: &str| -> Result<_, Box<dyn Error>> {
            Ok(dir.join(case[key].as_str().ok_or("Missing file name")?))
        };
        let raw = fs::read(file("packed")?)?;
        let activations = f32_file(&file("input")?)?;
        let reference = f32_file(&file("expected")?)?;
        let weight_reference = f32_file(&file("weights_f32")?)?;
        if activations.len() != m * k || reference.len() != m * n || weight_reference.len() != n * k
        {
            return Err("Fixture shapes do not match".into());
        }
        let upload = Instant::now();
        let packed = Q4kWeight::upload(&raw, n, k, &device)?;
        let input = Tensor::<B, 2>::from_data(TensorData::new(activations, [m, k]), &device);
        sync(&device)?;
        let upload_ms = upload.elapsed().as_secs_f64() * 1000.0;
        let first = Instant::now();
        let weights = packed.dequantize();
        sync(&device)?;
        let first_dequant_ms = first.elapsed().as_secs_f64() * 1000.0;
        let dequant_error = errors(
            &weights.clone().into_data().to_vec::<f32>()?,
            &weight_reference,
        )?;
        if dequant_error["normalized_max_error"].as_f64().unwrap() > 1e-6 {
            return Err(format!("Q4_K dequantization mismatch: {dequant_error}").into());
        }
        let weights_half = weights.clone().cast(FloatDType::F16);
        sync(&device)?;
        let mut variants = Vec::new();
        for variant in [
            "dequant_each_matmul",
            "cached_f32_matmul",
            "dequant_each_f16_matmul",
            "cached_f16_matmul",
        ] {
            let operation = || {
                let w = if variant.starts_with("dequant_each") {
                    packed.dequantize()
                } else if variant == "cached_f16_matmul" {
                    weights_half.clone()
                } else {
                    weights.clone()
                };
                if variant.contains("f16") {
                    input
                        .clone()
                        .cast(FloatDType::F16)
                        .matmul(w.cast(FloatDType::F16).transpose())
                        .cast(FloatDType::F32)
                } else {
                    input.clone().matmul(w.transpose())
                }
            };
            let warmup = Instant::now();
            for _ in 0..3 {
                let output = operation();
                sync(&device)?;
                drop(output);
            }
            let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
            let mut samples = Vec::new();
            for _ in 0..rounds {
                sync(&device)?;
                let start = Instant::now();
                let output = operation();
                sync(&device)?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let error = errors(&output.into_data().to_vec::<f32>()?, &reference)?;
                if error["relative_rmse"].as_f64().unwrap()
                    > manifest["max_relative_rmse"].as_f64().unwrap()
                    || error["normalized_max_error"].as_f64().unwrap()
                        > manifest["max_normalized_error"].as_f64().unwrap()
                {
                    return Err(format!("Matmul tolerance failed: {error}").into());
                }
                samples.push(json!({"elapsed_ms": elapsed_ms, "error": error}));
            }
            variants.push(json!({"variant":variant,"warmup_ms":warmup_ms,"samples":samples}));
        }
        cases.push(
            json!({"name":case["name"],"shape_mkn":[m,k,n],"packed_bytes":raw.len(),
            "prepared_payload_bytes":packed.prepared_bytes(),"expanded_weight_bytes":n*k*4,
            "expanded_f16_weight_bytes":n*k*2,
            "upload_ms":upload_ms,"first_dequant_ms":first_dequant_ms,"dequant_error":dequant_error,
            "variants":variants}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"backend":"Burn ROCm/CubeCL",
        "rounds":rounds,"warmup_per_variant":3,"model_sha256":manifest["model_sha256"],"cases":cases,
        "scope":"Q4_K scales prepared once on CPU; nibbles unpacked on GPU. Float weights materialized. FP16 variants include input F32->F16 and output F16->F32 casts. No routed MoE or full model. Synchronized host operation time excludes transfers."}))?
    );
    Ok(())
}
