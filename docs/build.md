# Build and hardware

[Back to README](../README.md)

Run these commands from the repository root. Install Rust 1.95+ and the ROCm/HIP SDK, and supply your own DiffusionGemma GGUF model. No C/C++ toolchain, CMake or git submodule is needed: the service runs DiffusionGemma with the Rust [CubeCL backend](cubecl.md), whose kernels compile at runtime through HIP.

```bash
cargo build --workspace --locked
```

## Hardware

The kernels target AMD RDNA3/RDNA3.5 GPUs (32-lane waves with WMMA matrix instructions). They are tested on the Radeon 8060S (`gfx1151`) under ROCm 7.2.1 on WSL2. NVIDIA and CPU inference are not supported.

GPU memory must hold the quantized weights (about 17.7 GB for Q4_K_M), the KV cache sized by `--context-size`, and about 1.1 GB for the optional vision projector. On APUs such as Strix Halo, GPU memory is system memory shared with everything else on the machine.

## ROCm/HIP

For the development machine's ROCm/WSL setup:

```bash
export ROCM_PATH=/opt/rocm-7.2.1
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export DIFFUSION_MODEL="$HOME/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
cargo run --release --locked -p jevons-rs -- --bind 127.0.0.1:8080
```

Adjust the SDK path and model path to your machine. Keep the runtime environment set in the shell that starts the service.

The first start on a GPU compiles kernels and measures launch plans (a few minutes on the test machine). Compiled kernels and plans are cached in `~/.cache/diffusion-cubecl` (override with `DIFFUSION_CUBECL_CACHE`), so later starts take about 35 s to load the weights. See [device tuning](cubecl.md#device-tuning).

## Runtime settings

Both inference binaries accept these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `-m`, `--model` | `DIFFUSION_MODEL` | Model GGUF file or Hugging Face checkpoint directory. |
| `--arch` | `auto` (`JEVONS_ARCH`) | Architecture: `auto`, `gemma4-diffusion` or `nemotron-diffusion`. `auto` detects it from the files; an explicit value must match them. |
| `--main-gpu` | `0` | Select the HIP device. |
| `--context-size` | `8192` | Limit prompt, thought framing/budget, and canvas tokens. |
| `--batch-size` | `512` | Limit each prefill chunk and the full canvas (at most 1024). |
| `--seed` | `42` | Seed the initial answer-slot noise. |
| `--no-prompt-cache` | Off | Recompute every prompt instead of reusing its cached prefix. |
| `--mmproj` | None | DiffusionGemma vision projector GGUF for image input (server only). |
| `--model-id` | Per architecture | Served model ID (server only), such as `gemmadiffusion-0.1`. |

## Image input

Supply a `gemma4v` vision-projector GGUF compatible with DiffusionGemma's Gemma 4 26B-A4B vision encoder and the text model's embedding width. The text GGUF alone cannot process images. Projectors and model weights are external assets and are never downloaded automatically.

Set `DIFFUSION_MMPROJ=/path/to/mmproj.gguf` or pass `--mmproj /path/to/mmproj.gguf` to the server. The CubeCL vision encoder resizes each image to 70–280 image tokens (48-pixel cells) and prefills it as one block that attends bidirectionally within itself, as in llama.cpp's DiffusionGemma integration. An image must fit in one `--batch-size` chunk.

Image decoder dependencies are Rust crates; WebP support does not require ffmpeg. See the [API examples](api.md#extensions) for payload formats and image limits.
