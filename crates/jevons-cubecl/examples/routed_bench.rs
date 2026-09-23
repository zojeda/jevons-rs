//! Complete grouped gate/up operator, including GPU grouping and output scatter.
use burn_rocm::RocmDevice;
use burn_tensor::{Tensor, TensorData, backend::Backend};
use jevons_cubecl::expert::{DirectBackend as D, PackedExpert, routed::Routes};
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
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
    if args.len() != 2 {
        return Err("Usage: routed_bench FIXTURES ROUNDS".into());
    }
    let dir = Path::new(&args[0]);
    let rounds: usize = args[1].parse()?;
    if !(1..=1000).contains(&rounds) {
        return Err("Rounds must be in 1..1000".into());
    }
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    let device = RocmDevice::new(0);
    let mut cases = Vec::new();
    for c in manifest["cases"].as_array().ok_or("Missing cases")? {
        let number = |key: &str| -> Result<usize> {
            Ok(c[key].as_u64().ok_or("Missing dimension")? as usize)
        };
        let (m, k, n, experts, top_k) = (
            number("m")?,
            number("k")?,
            number("n")?,
            number("experts")?,
            number("top_k")?,
        );
        let file =
            |key: &str| -> Result<_> { Ok(dir.join(c[key].as_str().ok_or("Missing file")?)) };
        let raw = fs::read(file("packed")?)?;
        let x = f32_file(&file("input")?)?;
        let reference = f32_file(&file("expected")?)?;
        let id_bytes = fs::read(file("ids")?)?;
        if !id_bytes.len().is_multiple_of(4) || x.len() != m * k || reference.len() != m * top_k * n
        {
            return Err("Fixture size mismatch".into());
        }
        let ids = id_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let routes = Routes::upload(ids, m, experts, top_k, &device)?;
        let packed = PackedExpert::upload(&raw, n * experts, k, &device)?;
        let input = Tensor::<D, 2>::from_data(TensorData::new(x, [m, k]), &device);
        sync(&device)?;
        let mut variants = Vec::new();
        for tokens in [4, 8] {
            let operation = || packed.matmul_routed(input.clone(), &routes, tokens, 4);
            let check = |out: Tensor<D, 3>| -> Result<Value> {
                let e = errors(&out.into_data().to_vec::<f32>()?, &reference)?;
                if e["relative_rmse"].as_f64().unwrap()
                    > manifest["max_relative_rmse"].as_f64().ok_or("Tolerance")?
                    || e["normalized_max_error"].as_f64().unwrap()
                        > manifest["max_normalized_error"]
                            .as_f64()
                            .ok_or("Tolerance")?
                {
                    return Err(format!("Routed numerical failure: {e}").into());
                }
                Ok(e)
            };
            let warmup = Instant::now();
            for _ in 0..3 {
                let out = operation()?;
                sync(&device)?;
                check(out)?;
            }
            let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
            let mut samples = Vec::new();
            for _ in 0..rounds {
                sync(&device)?;
                let start = Instant::now();
                let out = operation()?;
                sync(&device)?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                samples.push(json!({"elapsed_ms":elapsed_ms,"error":check(out)?}));
            }
            variants.push(json!({"variant":format!("grouped_q4k_tokens{tokens}_waves4"),"warmup_ms":warmup_ms,"samples":samples}));
            eprintln!("{} tokens{tokens} passed", c["name"]);
        }
        cases.push(json!({"name":c["name"],"shape_mkn":[m,k,n],"experts":experts,"top_k":top_k,"packed_bytes":raw.len(),"prepared_payload_bytes":n*k*experts/256*192,"expanded_f16_weight_bytes":null,"variants":variants}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"backend":"CubeCL grouped Q4_K", "rounds":rounds,"model_sha256":manifest["model_sha256"],"cases":cases,"scope":manifest["scope"]})
        )?
    );
    Ok(())
}
