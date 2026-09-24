# Burn 0.22 on ROCm: Phase 1 feasibility spike

Setup: Burn `=0.22.0-pre.4`, cubecl `=0.11.0-pre.4`, cubek `=0.3.0-pre.4`, ROCm 7.2.1, WSL2. The GPU is an AMD Radeon 8060S (`gfx1151`, 32-lane waves) with memory shared with the host. All runs used `--release`. The code is in `spikes/burn022` (`src/bin/{init,matmul,attn,kv,pick,store,tok}.rs`, plus `dual/` for item 8). Each crate has its own `[workspace]` and every file carries `#![forbid(unsafe_code)]`.

**Critical finding:** in cubecl 0.11, HIP compiles through a new pliron/LLVM backend by default. That backend **panics on BF16 kernels**: `cubecl-llvm/src/shared/to_llvm/constant.rs:72`, `float_attr(..).unwrap()`, hit by the first fused bf16 elementwise op. The fix is to enable the `cpp` feature on `cubecl-hip`, which restores the HIP C++/hiprtc path used in 0.10. Every result below was measured with `cpp` enabled:

```toml
cubecl-hip = { version = "=0.11.0-pre.4", default-features = false, features = ["cpp"] }
```

## Results

| # | Item | Result | Numbers | Go/No-go | Notes |
|---|------|--------|---------|----------|-------|
| 1 | HIP init, cache, memory release | Works | First tensor + sync 0.19–0.56 s. First fused op 14.8 s cold, 24 ms with the cache warm | **Go** (only with `cubecl-hip/cpp`) | Device: `Device::rocm(0)`, which is `DispatchDevice::Cube(cubecl::Device::Hip(AmdDevice(0)))`. The compile and autotune cache is a SQLite "environment" (`default.db`), set with `config.environment.path = CacheConfig::Directory(..)` plus `compilation.cache = true`. It is also read from `cubecl.toml` or the `[cubecl]` section of `burn.toml`, and the env var `CUBECL_ENVIRONMENT` selects the environment name. Dropped tensors return to the pool but stay reserved until `device.memory_cleanup()`: 2 × 1 GiB live reserved **7.9 GB**, and cleanup brought it back to 31 MB. Weights should go through `memory_persistent_allocations`, which reserves exactly the size (32 MiB used 32 MiB). `device.memory_pool_usage()` reports `{number_allocs, bytes_in_use, bytes_reserved}`. On this APU, device allocations also show up in process RSS. |
| 2 | BF16 matmul | Correct. Throughput is mediocre for small M | Max error about 1 bf16 ulp of the output (0.24 at rms 21, K=4096). Timings in the table below | **Go** for prefill. Decode needs a custom GEMV | Odd N=131073 works at the same speed as even N. A `[N,K]` weight used as `w.transpose()` (no copy) is as fast as or faster than a contiguous `[K,N]` weight for M ≥ 128. **Row invariance:** M=1 takes a different path, and 8 of 4096 elements differ from larger M by 1 ulp. M=2…2048 produced bitwise-identical rows, and repeated runs at the same M are bitwise identical. Autotune has no off switch (`CUBECL_AUTOTUNE_LEVEL=minimal` is the lowest level), and its choices are cached per shape. For guaranteed invariance, call `burn_cubecl::kernel::matmul::matmul(.., MatmulStrategy::Cube, ..)` from an extension op, or use a custom kernel. |
| 3 | Attention | Works. GQA needs a reshape trick | Timings in the table below. bf16 vs f32 reference: max error 1.4e-3 (L=32, S=4096) and 6.8e-3 (L=512, S=4096) | **Go** | Call: `burn::tensor::module::attention(q,k,v,mask,bias,opts)`. **Passing 32 q heads with 8 kv heads is not rejected. It runs and returns wrong results (max error 3.0).** Guard against it. Correct ways to do GQA without repeating KV: (a) **fold** q `[1,32,L,D]` → `[1,8,4L,D]`, which is exact without a mask (a masked version is shown below), or (b) broadcast through the batch dim, q `[8,4,L,D]` with k/v `[8,1,S,D]`. `is_causal` with q_len < k_len is **bottom-right aligned** in both the flash kernel and the fallback. With folding, causality must be expressed as an explicit bool mask, `true` = masked, and `[1,1,..]` expanded to `[1,8,..]` works. A mask costs 0–13%. Setting `scale`, `softcap` or `attn_bias` forces the slow fallback, so pre-scale q instead. The flash tiles use f16 fragments. |
| 4 | KV cache update | In place when the tensor is the sole owner | Cap 8192 (16 MiB): 0.22 ms per update including a sync, flat for 1, 32 and 512 new positions. Cap 65536 (128 MiB): 0.23–0.30 ms, also flat, while a full copy takes 1.68 ms | **Go** | `slab = slab.slice_assign([0..1,0..8,p..p+n,0..128], new)` runs in place when no other handle exists: the cost does not grow with cap. With a clone alive it silently copies the whole slab (1.3–1.7 ms at 128 MiB). Only tested with fusion on; a non-fusion build was skipped to avoid another feature combination. The ~0.2 ms floor is launch plus `sync`. Plain elementwise bandwidth is **188 GB/s** (256 MiB read+write). |
| 5 | Custom kernel as a Burn op | Works with no `unsafe` | pick-logits, 2048 pairs, d=4096: 0.050 ms. Max error vs a CPU f32 reference is 8.9e-7. Composes with fused ops before and after (difference 0) | **Go** | `#[backend_extension(Cube, Fusion)]` on a trait, implemented for `burn_cubecl::CubeBackend`, called through `Dispatch::op(t.into_dispatch())` and wrapped with `Tensor::from_dispatch`. The macro emits `unsafe` only when the trait method is declared `unsafe`. `#[cube(launch)]` generates a safe, checked `launch`; only `launch_unchecked` is unsafe. The whole bin compiles under `#![forbid(unsafe_code)]`. Checked launch works directly on `CubeTensor::into_tensor_arg()`. |
| 6 | Streaming safetensors | Works tensor by tensor, BF16 kept end to end | gate_proj `[14336,4096]` (112 MiB): burn-store 240 ms, peak RSS 468 MiB; manual 182 ms, peak 357 MiB. embed_tokens `[131073,4096]` (1 GiB): burn-store 1.76 s, peak **3.2 GB**; manual 1.30 s, peak **2.2 GB** | **Go** (manual reader preferred) | `SafetensorsStore::from_file(..).get_tensor(name)` mmaps the file (the `unsafe` stays inside burn-store) and copies one tensor lazily, giving `burn_pack::Tensor` then `to_bytes()` then `TensorData::from_bytes`. The index of 286 tensors builds in 8 ms. The manual reader parses the header JSON and does a positioned `read_exact`, with no mmap and no unsafe; its peak is roughly host copy plus upload staging. Values match on the device. The vocab is **131073** rows, so odd N is real. |
| 7 | `tokenizers` 0.23.2, `default-features=false`, `fancy-regex` | Works, pure Rust | Loads in 286 ms. Encodes 17 k chars (6200 tokens) in 2.9 ms | **Go** | No onig and no `cc`: `esaxx-rs` is built without its `cpp` feature. `"<|im_start|>user\nHi<|im_end|>"` → `[10, 3263, 1010, 37133, 11]`. `add_special_tokens` adds nothing (no BOS from the post-processor). **Literal mode:** build a second tokenizer from the same JSON with `added_tokens = []`. `set_encode_special_tokens(true)` is not enough, because `<think>`, `</think>`, `<tool_call>` and `<|image_*|>` are added with `special=false` and still map to 12–21. With the literal tokenizer, `"</think>"` → `[1885, 74045, 1062]` and `"<|im_start|>"` → `[1060, 1124, 1329, 18993, 1124, 1062]`, with no id ≤ 1000 in any sample. Plain text gives identical ids with both tokenizers, and decoding round-trips. Build prompts by splicing control ids from the full tokenizer around user text from the literal one. |
| 8 | cubecl 0.10 + 0.11 in one binary | Works | Both runtimes init HIP and run kernels in one process in either order, interleaved, with both clients alive | **Go** | Both versions share `cubecl-hip-sys 7.14.6085001`, so there is no `links` conflict. `#[cube]` expands to `cubecl::…` paths, so each kernel module needs `use cubecl010 as cubecl;` (or `cubecl011`). |

### Matmul throughput (item 2)

Measured with the weight stored as `[N,K]` and used as `w.transpose()`, with the autotune cache warm.

| Shape | Time | Throughput |
|-------|------|------------|
| `[512x4096]x[4096x14336]` | 6.8–8.5 ms | 7–8.8 TFLOP/s |
| `[32x4096]x[4096x131072]` | 25 ms (19 ms with contiguous `[K,N]`) | 43 GB/s |
| `[32x4096]x[4096x131073]` | 25 ms (15–19 ms with contiguous `[K,N]`) | about the same as even N |
| `[1x4096]x[4096x4096]` | 0.71 ms | **47 GB/s**, against 188 GB/s attainable |
| `[32x4096]x[4096x4096]` | 0.61 ms | |
| `[128x4096]x[4096x4096]` | 0.75–1.3 ms | |
| `[512x4096]x[4096x4096]` | 2.1–2.4 ms | 7–8 TFLOP/s |
| `[2048x4096]x[4096x4096]` | 6.4–7.9 ms | 8.7–10.7 TFLOP/s |

On a cold cache the first call to a new shape costs 0.1–12 s of autotuning; M=1 took 10–12 s.

### Attention timing (item 3)

bf16, 32 q heads, 8 kv heads, head_dim 128.

| q_len, k_len | Fold-GQA | Fold + bool mask | Repeated KV (materialized) |
|--------------|----------|------------------|----------------------------|
| L=32, S=1024 | 0.69 ms | 0.69 ms | 1.10 ms |
| L=32, S=4096 | 1.77 ms | 1.61 ms | 3.55 ms |
| L=32, S=8192 | 3.33 ms | 3.21 ms | 6.01 ms |
| L=512, S=4096 | 16.6 ms | 18.8 ms | 18.9 ms |
| L=2048, S=2048 | 27.4 ms | 30.1 ms | 27.8 ms (30.7 ms with `is_causal`) |

## Recommendations for Phase 2

- Enable `cubecl-hip/cpp`. Point the environment cache at `~/.cache/jevons-…` and warm it once; cold autotune takes minutes.
- Load weights with `memory_persistent_allocations` and a manual safetensors reader (positioned reads, no unsafe). Call `memory_cleanup()` after prefill spikes.
- Keep Burn matmul for prefill and batched work. Write custom CubeCL GEMV/pick kernels as backend extensions for decode-sized M, where Burn reaches only 47 GB/s against 188 GB/s attainable.
- Attention: fold GQA into the sequence dim. Block-diffusion queries see the whole prefix and their own block, so they need no mask. Add a runtime assertion that q and kv heads match, because the kernel does not check it.
- KV cache: preallocate the slab, keep a single owner, and update it with `slice_assign`.
- Row invariance across batch sizes is not guaranteed by autotune (M=1 differs). If reproducibility across M matters, force one strategy or use custom kernels.

## Working 0.22 API patterns

Dependencies:

```toml
burn = { version = "=0.22.0-pre.4", default-features = false, features = ["std", "rocm", "fusion", "autotune", "extension", "safetensors"] }
burn-cubecl = { version = "=0.22.0-pre.4", default-features = false, features = ["std", "hip", "fusion", "autotune"] }  # for extension impls
cubecl = { version = "=0.11.0-pre.4", features = ["hip"] }
cubecl-hip = { version = "=0.11.0-pre.4", default-features = false, features = ["cpp"] }        # REQUIRED for bf16
tokenizers = { version = "0.23.2", default-features = false, features = ["fancy-regex"] }
```

Device, cache, memory:

```rust
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
let mut cfg = CubeClRuntimeConfig::from_current_dir().override_from_env();
cfg.compilation.cache = true;
cfg.environment.path = CacheConfig::Directory(cache_dir);
CubeClRuntimeConfig::set(cfg);                       // before first device use
let device = burn::tensor::Device::rocm(0);          // Device<Cube(Hip(AmdDevice(0)))>
device.identity();                                    // name "AMD Radeon(TM) 8060S Graphics", fingerprint "hip-kernel_gfx1151"
device.memory_pool_usage();  device.memory_cleanup();  device.sync()?;
```

Tensor from BF16 data, matmul:

```rust
use burn::tensor::{DType, Tensor, TensorData};
let w: Tensor<2> = device.memory_persistent_allocations(data, |d| Tensor::from_data(d, (&device, DType::BF16)));
let x = Tensor::<2>::from_data(TensorData::new(vec_bf16, [m, k]), (&device, DType::BF16));
let y = x.matmul(w.transpose());                      // w stored [n, k]; no copy
let host: Vec<half::bf16> = y.into_data().try_to_vec::<half::bf16>()?;
```

Attention (GQA fold, no repeated KV):

```rust
use burn::tensor::{module::attention, ops::AttentionModuleOptions};
// q [1,32,L,128] -> [1,8,4L,128] (q heads h = kv*4+g are contiguous per kv head)
let o = attention(q.reshape([1, 8, 4 * l, 128]), k, v, None /* or Some(bool mask [1,8,4L,S], true=masked) */,
                  None, AttentionModuleOptions::default()).reshape([1, 32, l, 128]);
// is_causal: bottom-right aligned (q row i sees keys <= i + S - L); only valid with matching head counts.
```

KV slab update (in place when this is the only handle):

```rust
let mut slab = Tensor::<4>::zeros([1, 8, cap, 128], (&device, DType::BF16));
slab = slab.slice_assign([0..1, 0..8, pos..pos + n, 0..128], new_kv);
```

Custom kernel op (checked launch, no unsafe):

```rust
use burn::backend::{Backend, Dispatch, backend_extension, cubecl::dtype_to_storage_type, tensor::{FloatTensor, IntTensor}};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, ops::numeric::empty_device_dtype};

mod kern { use cubecl::prelude::*;
  #[cube(launch)]
  pub fn pick_kernel<F: Float, I: Int>(hidden: &Tensor<F>, table: &Tensor<F>, pairs: &Tensor<I>,
      out: &mut Tensor<f32>, #[comptime] d: usize, #[define(F, I)] _dt: [ElemType; 2]) {
      let p = CUBE_POS_X as usize;
      let (row, tok) = (usize::cast_from(pairs[2 * p]), usize::cast_from(pairs[2 * p + 1]));
      let mut scratch = Shared::<[f32]>::new_slice(8usize);
      let mut acc = 0.0f32;
      #[unroll] for i in 0..comptime!(d / 256) { let c = i * 256 + UNIT_POS_X as usize;
          acc += f32::cast_from(hidden[row * d + c]) * f32::cast_from(table[tok * d + c]); }
      let s = plane_sum(acc);
      if UNIT_POS_X % 32 == 0 { scratch[(UNIT_POS_X / 32) as usize] = s; }
      sync_cube();
      if UNIT_POS_X == 0 { let mut t = 0.0f32; #[unroll] for w in 0..8usize { t += scratch[w]; } out[p] = t; }
  } }

#[backend_extension(Cube, Fusion)]
pub trait PickOps: Backend {
    #[fusion(dtype = { DType::F32 }, shape = { Shape::new([pairs[0]]) })]
    fn pick_logits(hidden: FloatTensor<Self>, table: FloatTensor<Self>, pairs: IntTensor<Self>) -> FloatTensor<Self>;
}
impl PickOps for CubeBackend {
    fn pick_logits(h: FloatTensor<Self>, t: FloatTensor<Self>, p: IntTensor<Self>) -> FloatTensor<Self> {
        let (h, t, p) = (into_contiguous(h), into_contiguous(t), into_contiguous(p));
        let (d, n) = (h.meta.shape[1], p.meta.shape[0]);
        let out = empty_device_dtype(h.client.clone(), h.device.clone(), Shape::new([n]), DType::F32);
        let dts = [dtype_to_storage_type(h.dtype), dtype_to_storage_type(p.dtype)];
        let client = h.client.clone();
        kern::pick_kernel::launch(&client, CubeCount::Static(n as u32, 1, 1), CubeDim::new_1d(256),
            h.into_tensor_arg(), t.into_tensor_arg(), p.into_tensor_arg(), out.clone().into_tensor_arg(), d, dts);
        out
    }
}
pub fn pick_logits(h: Tensor<2>, t: Tensor<2>, p: Tensor<2, Int>) -> Tensor<1> {
    Tensor::from_dispatch(Dispatch::pick_logits(h.into_dispatch(), t.into_dispatch(), p.into_dispatch()))
}
```

Streaming safetensors:

```rust
// burn-store (mmap inside burn-store, lazy per-tensor copy)
let mut store = burn::store::SafetensorsStore::from_file(path);
let pt = store.get_tensor("encoder.embed_tokens.weight")?.unwrap();  // burn_pack::Tensor {dtype, shape}
let data = TensorData::from_bytes(pt.to_bytes()?, pt.shape.clone(), pt.dtype);
// manual (no mmap/unsafe, lower peak RSS): read u64 header len, serde_json header,
// seek to 8 + hdr + data_offsets.0, read_exact, then TensorData::from_bytes_vec(bytes, shape, DType::BF16)
```

Tokenizer, full and literal:

```rust
let json = std::fs::read_to_string(path)?;
let full: tokenizers::Tokenizer = json.parse()?;
let mut v: serde_json::Value = serde_json::from_str(&json)?;
v["added_tokens"] = serde_json::Value::Array(vec![]);
let literal: tokenizers::Tokenizer = v.to_string().parse()?;   // user text: never yields ids 0..=1000
```
