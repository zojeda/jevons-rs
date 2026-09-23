//! Controlled comparison of autotuned FP16, explicit tiles, and fused Q4_K.
use burn_rocm::RocmDevice;
use burn_tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use jevons_cubecl::{
    expert::{DirectBackend as D, PackedExpert, TileConfig, explicit_f16},
    q4k::HipBackend as B,
};
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Clone, Copy)]
enum PackedConfig {
    Scalar { tokens: usize, waves: u32 },
    Cmma { stage_k: usize, waves: u32 },
}
fn f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("Invalid f32 bytes".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect())
}
fn errors(actual: &[f32], reference: &[f32]) -> Result<Value> {
    if actual.len() != reference.len() || actual.is_empty() {
        return Err("Invalid output shape".into());
    }
    let (mut sq, mut norm, mut max, mut peak) = (0.0, 0.0, 0.0_f64, 0.0_f64);
    for (&a, &b) in actual.iter().zip(reference) {
        if !a.is_finite() || !b.is_finite() {
            return Err("Nonfinite output".into());
        }
        let e = (f64::from(a) - f64::from(b)).abs();
        sq += e * e;
        norm += f64::from(b).powi(2);
        max = max.max(e);
        peak = peak.max(f64::from(b).abs());
    }
    Ok(
        json!({"relative_rmse":(sq/norm.max(1e-30)).sqrt(),"normalized_max_error":max/peak.max(1e-30),"max_abs_error":max}),
    )
}
fn sync(device: &RocmDevice) -> Result<()> {
    D::sync(device).map_err(|e| format!("{e:?}").into())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("Usage: expert_tune FIXTURES ROUNDS [VARIANT,VARIANT,...]".into());
    }
    let rounds: usize = args[1].parse()?;
    if !(1..=1000).contains(&rounds) {
        return Err("Rounds must be in 1..1000".into());
    }
    let dir = Path::new(&args[0]);
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    let device = RocmDevice::new(0);
    let mut cases = Vec::new();
    for case in manifest["cases"].as_array().ok_or("Missing cases")? {
        let m = case["m"].as_u64().ok_or("m")? as usize;
        let k = case["k"].as_u64().ok_or("k")? as usize;
        let n = case["n"].as_u64().ok_or("n")? as usize;
        let file = |key: &str| -> Result<_> { Ok(dir.join(case[key].as_str().ok_or("file")?)) };
        let raw = fs::read(file("packed")?)?;
        let x = f32_file(&file("input")?)?;
        let w = f32_file(&file("weights_f32")?)?;
        let reference = f32_file(&file("expected")?)?;
        if x.len() != m * k || w.len() != n * k || reference.len() != m * n {
            return Err("Fixture shape mismatch".into());
        }
        let input = Tensor::<D, 2>::from_data(TensorData::new(x.clone(), [m, k]), &device);
        let half = Tensor::<D, 2>::from_data(TensorData::new(w.clone(), [n, k]), &device)
            .cast(FloatDType::F16);
        let control_x = Tensor::<B, 2>::from_data(TensorData::new(x, [m, k]), &device);
        let control_w =
            Tensor::<B, 2>::from_data(TensorData::new(w, [n, k]), &device).cast(FloatDType::F16);
        let packed = PackedExpert::upload(&raw, n, k, &device)?;
        B::sync(&device).map_err(|e| format!("{e:?}"))?;
        sync(&device)?;
        let mut variants = vec![("cached_f16_matmul".to_owned(), None, None)];
        for (pm, pn, pk, planes) in [
            (1, 1, 2, 1),
            (1, 2, 2, 2),
            (1, 2, 4, 4),
            (1, 4, 4, 4),
            (2, 2, 2, 2),
            (2, 4, 4, 4),
        ] {
            variants.push((
                format!("tile_{pm}_{pn}_{pk}_planes{planes}"),
                Some(TileConfig {
                    partition_m: pm,
                    partition_n: pn,
                    partition_k: pk,
                    planes,
                }),
                None,
            ));
        }
        for tokens in [1, 2, 4, 8] {
            for waves in [4, 8] {
                variants.push((
                    format!("fused_q4k_tokens{tokens}_waves{waves}"),
                    None,
                    Some(PackedConfig::Scalar { tokens, waves }),
                ));
            }
        }
        for stage_k in [32, 64] {
            for waves in [2, 4, 8] {
                variants.push((
                    format!("fused_cmma_k{stage_k}_waves{waves}"),
                    None,
                    Some(PackedConfig::Cmma { stage_k, waves }),
                ));
            }
        }
        if let Some(filter) = args.get(2) {
            let names: Vec<_> = filter.split(',').collect();
            if names
                .iter()
                .any(|name| !variants.iter().any(|v| &v.0 == name))
            {
                return Err("Unknown variant in filter".into());
            }
            variants.retain(|v| names.contains(&v.0.as_str()));
        }
        let mut measured = Vec::new();
        for (name, tile, fused) in variants {
            // A common output enum preserves the original fusion-enabled control.
            enum Output {
                Control(Tensor<B, 2>),
                Direct(Tensor<D, 2>),
            }
            let operation = || -> Result<Output> {
                if name == "cached_f16_matmul" {
                    return Ok(Output::Control(
                        control_x
                            .clone()
                            .cast(FloatDType::F16)
                            .matmul(control_w.clone().transpose())
                            .cast(FloatDType::F32),
                    ));
                }
                let out = if let Some(config) = tile {
                    explicit_f16(input.clone(), half.clone(), config)?
                } else if let Some(config) = fused {
                    match config {
                        PackedConfig::Scalar { tokens, waves } => {
                            packed.matmul(input.clone(), tokens, waves)?
                        }
                        PackedConfig::Cmma { stage_k, waves } => {
                            packed.matmul_cmma(input.clone(), waves, stage_k)?
                        }
                    }
                } else {
                    return Err("Missing experiment configuration".into());
                };
                Ok(Output::Direct(out))
            };
            let synchronize = || -> Result<()> {
                if name == "cached_f16_matmul" {
                    B::sync(&device).map_err(|e| format!("{e:?}").into())
                } else {
                    sync(&device)
                }
            };
            let check = |out: Output| -> Result<Value> {
                let values = match out {
                    Output::Control(t) => t.into_data().to_vec::<f32>()?,
                    Output::Direct(t) => t.into_data().to_vec::<f32>()?,
                };
                let e = errors(&values, &reference)?;
                if e["relative_rmse"].as_f64().unwrap()
                    > manifest["max_relative_rmse"].as_f64().ok_or("tolerance")?
                    || e["normalized_max_error"].as_f64().unwrap()
                        > manifest["max_normalized_error"]
                            .as_f64()
                            .ok_or("tolerance")?
                {
                    return Err(format!("Numerical check failed for {name}: {e}").into());
                }
                Ok(e)
            };
            let start = Instant::now();
            // Unsupported configurations are recorded; numerical failures stop the run.
            let first = match operation() {
                Ok(out) => out,
                Err(error) => {
                    measured.push(json!({"variant":name,"unavailable":error.to_string()}));
                    continue;
                }
            };
            synchronize()?;
            check(first)?;
            for _ in 0..2 {
                let out = operation()?;
                synchronize()?;
                check(out)?;
            }
            let warmup_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut samples = Vec::new();
            for _ in 0..rounds {
                synchronize()?;
                let start = Instant::now();
                let out = operation()?;
                synchronize()?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                samples.push(json!({"elapsed_ms":elapsed_ms,"error":check(out)?}));
            }
            eprintln!("{} {name} passed", case["name"]);
            measured.push(json!({"variant":name,"warmup_ms":warmup_ms,"samples":samples}));
        }
        cases.push(json!({"name":case["name"],"shape_mkn":[m,k,n],"packed_bytes":raw.len(),"prepared_payload_bytes":n*k/256*192,"expanded_f16_weight_bytes":n*k*2,"variants":measured}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"backend":"CubeCL explicit expert kernels","rounds":rounds,"warmup_per_variant":3,"model_sha256":manifest["model_sha256"],"cases":cases,"scope":"Isolated experts; F16 variants include F32/F16 conversions; fused Q4_K uses F32 input, accumulation and output. Full dequantized weights are not materialized by the fused operation. Host synchronized compute excludes upload/readback. No routing."})
        )?
    );
    Ok(())
}
