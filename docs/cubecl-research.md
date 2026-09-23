# CubeCL migration research

Research date: 2026-09-21. These findings distinguish verified upstream capabilities from migration recommendations. Links to `main` are moving references; pin actual dependency versions and record their commits before implementation.

## What CubeCL replaces

CubeCL supplies a Rust kernel language, JIT compiler, intermediate representation, and execution runtimes. It is lower level than a model inference framework. Its documentation positions Burn above it. Thus replacing llama.cpp with CubeCL means implementing or adopting the model execution layer, rather than exchanging equivalent inference APIs. CubeCL's public API remains alpha and upstream recommends version pinning. [CubeCL README](https://github.com/tracel-ai/cubecl#readme)

Burn supplies tensor operations, neural-network abstractions and automatic fusion. Its ROCm backend explicitly depends on CubeCL with the `hip` feature. This makes **a DiffusionGemma model implemented with Burn tensors, running on CubeCL**, a reasonable first architecture to investigate. This recommendation does not imply Burn already implements the required model. [Burn](https://github.com/tracel-ai/burn#readme), [burn-rocm manifest](https://github.com/tracel-ai/burn/blob/main/crates/burn-rocm/Cargo.toml)

CubeK provides reusable CubeCL matrix multiplication, attention, reductions, random-number and quantization kernels. This reduces the direct-CubeCL kernel workload but does not supply an entire model runtime. Its documented quantization families include q2, q4, q8 and fp4; those names alone do not establish binary compatibility with any particular GGUF quantization format. [CubeK algorithms](https://github.com/tracel-ai/cubek#algorithms)

## Platform and dependency constraints

CubeCL documents CUDA for NVIDIA, HIP/ROCm for AMD, wgpu paths for Metal/Vulkan/WebGPU, and a CPU runtime. Feature and instruction availability differs across targets; unsupported instructions can fail during runtime compilation. Advertised backend support is therefore a starting point for a hardware smoke test, not proof the complete model will work. [CubeCL supported platforms](https://github.com/tracel-ai/cubecl#supported-platforms)

The HIP runtime specifically describes hardware-dependent matrix acceleration and a rocWMMA requirement for some architectures. Validate against the actual GPU, ROCm installation and WSL environment before spending effort on the model port. [HIP runtime README](https://github.com/tracel-ai/cubecl/blob/main/crates/cubecl-hip/README.md)

A verified release combination is **Burn 0.21.0 → CubeCL 0.10.0 / CubeK 0.2.0**. CubeCL 0.10.0 declares Rust **1.92**, above this repository's Rust 1.88 minimum. The current CubeCL main snapshot declares 0.11.0-pre.4 and Rust 1.95; Burn main is 0.22.0-pre.3 and pins a CubeCL Git revision. Start by evaluating the release combination rather than following both main branches independently. This is a compatibility recommendation, not a claim that the release combination meets all needed model operations. [Burn 0.21 manifest](https://github.com/tracel-ai/burn/blob/v0.21.0/Cargo.toml), [CubeCL 0.10 manifest](https://github.com/tracel-ai/cubecl/blob/v0.10.0/Cargo.toml), [CubeCL main manifest](https://github.com/tracel-ai/cubecl/blob/main/Cargo.toml), [Burn main manifest](https://github.com/tracel-ai/burn/blob/main/Cargo.toml)

## Model, weights and tokenization gaps

The inspected official Tracel models catalog does not list DiffusionGemma. That is evidence of an unverified implementation gap, not proof no external implementation exists. Do not assume generic Gemma or autoregressive Llama implementations provide DiffusionGemma semantics. [Tracel model catalog](https://github.com/tracel-ai/models#readme)

The maintained Hugging Face Diffusers implementation describes a causal encoder cache and a bidirectional diffusion decoder operating on token canvases. Its model is `DiffusionGemmaForBlockDiffusion` in Transformers. The migration must reproduce the exact operations exercised by this repository's pinned native implementation, including its scoring/probing behavior, rather than replacing that behavior with ordinary next-token generation. Hugging Face is an additional architecture reference; the repository's existing implementation should remain the behavioral oracle. [Diffusers DiffusionGemma documentation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/diffusion_gemma.md)

Burn Store documents SafeTensors, PyTorch and Burnpack loading plus tensor remapping; the inspected format list does not advertise GGUF. Therefore plan explicit GGUF validation/conversion work or use independently supplied compatible SafeTensors weights. Loading weight tensors does not construct the required model architecture. Quantized weight layouts, tensor names, transpositions, scale interpretation and tied weights require validation. [Burn Store](https://github.com/tracel-ai/burn/blob/main/crates/burn-store/README.md)

Recommended tokenizer acceptance criterion: byte-identical token IDs against the existing engine for special tokens, whitespace, escaping, Unicode and all compiled question templates. Select a Rust tokenizer only after identifying the exact tokenizer assets/metadata present locally. Avoid keeping an implicit llama.cpp dependency merely for tokenization in the claimed replacement build.

## Architecture options and recommended gates

| Option | Main benefit | Work retained by this project |
| --- | --- | --- |
| Burn over CubeCL, with targeted custom CubeCL kernels | Reuse tensors, layers, storage and fusion | Model graph, checkpoint mapping, tokenizer, cache and diffusion-specific execution; any unsupported operations |
| Direct CubeCL plus CubeK | Maximum control over execution and memory | All of the above plus tensor orchestration, device memory planning and broader kernel integration |

Recommendation: begin with Burn over CubeCL as a feasibility spike. A direct implementation becomes justified if measured model requirements or kernel integration constraints prevent an acceptable Burn path.

1. Inventory the exact model checkpoint, GGUF tensor types, device memory, Rust toolchain and current baseline outputs. Freeze a reproducible existing-engine reference.
2. Compile and run a small tensor/kernel calculation on the intended HIP device with pinned dependencies; verify readback and numerical agreement. CPU support may have separate toolchain dependencies and should be checked independently.
3. Prove tokenizer agreement and load a representative weight tensor with known shape/value checks. Choose a memory-feasible weight strategy before loading the full model. Dequantization can increase memory substantially.
4. Implement the smallest complete forward/scoring path needed by existing structured inference. Compare intermediate activations and logits before integrating HTTP serving; final generated text alone is an insufficient correctness test.
5. Preserve existing request validation, scheduling, bounded queue, reproducibility and response mapping behind an engine boundary. Keep the current native backend available for differential tests until acceptance gates pass.
6. Measure correctness, peak host/device memory, cold initialization/JIT cost and warm latency on the same workload. Set numeric error tolerances from baseline measurements rather than inventing universal tolerances across differing precision and quantization.

Unresolved feasibility questions: the actual checkpoint's quantization formats; full weight/cache/activation memory budget; availability of every DiffusionGemma operation on the selected version/backend; exact HIP/WSL compatibility; tokenizer assets; and numerical equivalence required for this application's probability outputs. These should be resolved before estimating the complete engine rewrite.
