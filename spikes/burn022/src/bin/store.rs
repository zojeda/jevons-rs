#![forbid(unsafe_code)]
//! Item 6: stream individual BF16 tensors from safetensors onto the ROCm device.
//! usage: store <burnstore|manual> <tensor-name> <file>
use burn::store::{ModuleStore, SafetensorsStore};
use burn::tensor::{DType, Tensor, TensorData};
use burn022_spike::*;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

/// Header-only parse + positioned read of one tensor: no mmap, no unsafe, RSS ~= tensor size.
fn manual_read(path: &str, name: &str) -> TensorData {
    let mut f = std::fs::File::open(path).unwrap();
    let mut n = [0u8; 8];
    f.read_exact(&mut n).unwrap();
    let n = u64::from_le_bytes(n) as usize;
    let mut header = vec![0u8; 8 + n];
    f.seek(SeekFrom::Start(0)).unwrap();
    f.read_exact(&mut header).unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&header[8..]).unwrap();
    let info = &meta[name];
    assert_eq!(info["dtype"], "BF16");
    let shape: Vec<usize> = info["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let off = info["data_offsets"].as_array().unwrap();
    let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
    let mut bytes = vec![0u8; e - s];
    f.seek(SeekFrom::Start((8 + n + s) as u64)).unwrap();
    f.read_exact(&mut bytes).unwrap();
    TensorData::from_bytes_vec(bytes, shape, DType::BF16)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, name, file) = (&args[1], &args[2], &args[3]);
    let device = rocm();
    let _ = Tensor::<1>::zeros([1], &device); // init device before measuring
    device.sync().unwrap();
    println!("baseline RSS {} MiB, HWM {} MiB", vm_rss_kb() / 1024, vm_hwm_kb() / 1024);
    let t = Instant::now();
    let data = match mode.as_str() {
        "burnstore" => {
            let mut store = SafetensorsStore::from_file(file);
            let t_keys = Instant::now();
            let nkeys = store.keys().unwrap().len();
            println!("burn-store index of {nkeys} tensors: {:.1} ms, RSS {} MiB", t_keys.elapsed().as_secs_f64() * 1e3, vm_rss_kb() / 1024);
            let pt = store.get_tensor(name).unwrap().expect("tensor");
            println!("  {name}: dtype {:?} shape {:?} bytes {}", pt.dtype, pt.shape, pt.byte_len());
            let (dtype, shape) = (pt.dtype, pt.shape.clone());
            let bytes = pt.to_bytes().unwrap();
            TensorData::from_bytes(bytes, shape, dtype)
        }
        "manual" => manual_read(file, name),
        _ => panic!("mode"),
    };
    let t_read = t.elapsed().as_secs_f64() * 1e3;
    println!("read to host: {t_read:.0} ms, dtype {:?} shape {:?}, RSS {} MiB HWM {} MiB", data.dtype, data.shape, vm_rss_kb() / 1024, vm_hwm_kb() / 1024);
    let first: Vec<f32> = data.as_slice::<half::bf16>().unwrap()[..4].iter().map(|x| x.to_f32()).collect();
    let t2 = Instant::now();
    let shape = data.shape.clone();
    let w: Tensor<2> = device.memory_persistent_allocations(data, |data| Tensor::<2>::from_data(data, (&device, DType::BF16)));
    device.sync().unwrap();
    println!("upload to device: {:.0} ms; total {:.0} ms; tensor dtype on device {:?} dims {:?} (file shape {:?})", t2.elapsed().as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3, w.dtype(), w.dims(), shape);
    let dev_first: Vec<f32> = w.clone().slice([0..1, 0..4]).cast(DType::F32).into_data().try_to_vec::<f32>().unwrap();
    println!("first values host {first:?} device {dev_first:?}");
    mem(&device, "after upload");
    println!("final RSS {} MiB, peak HWM {} MiB", vm_rss_kb() / 1024, vm_hwm_kb() / 1024);
}
