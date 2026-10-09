# DiffusionGemma on Burn: proposal

[Design](design.md) · [Tasks](tasks.md) · [Report](report.md) · [Issue #5](https://github.com/zojeda/jevons-rs/issues/5)

## Why

`jevons-gemma4-diffusion` is the only model runtime that does not run on Burn. It launches its
own CubeCL kernels on its own buffers; Nemotron-Labs-Diffusion and Parakeet run on Burn through
`jevons-burn`, on the same CubeCL runtime and device. One framework would mean shared layers,
weight streaming and attention, and a shorter way to Burn's wgpu backend.

DiffusionGemma is also the fastest and most accurate model here, and its speed comes from those
kernels. The question is whether it can move without paying for it.

## What changes

Nothing that ships, until the spike says so. The spike runs on the branch `spike/gemma-burn` and
answers the questions of issue #5 with measurements:

- what a tuned kernel costs when Burn calls it;
- one decoder layer on Burn against the current path, for parity and for time;
- which Burn extensions and kernels a full port needs;
- what runs on Burn's wgpu backend.

Its deliverable is `report.md` in this folder, with a recommendation: port, port in part, or keep
the current runtime. The spike's code stays on its branch.

## The rule: no step back in speed

A component moves to Burn only when all of this holds on the machine the benchmarks use:

- **Time.** Measured in one process and interleaved with the current path, its median is within
  the current path's own spread from run to run, at canvas size (32 rows) and at prefill size
  (512 rows).
- **Bits.** Prompt rows stay bitwise identical however the prompt is chunked, so prefix reuse
  keeps its promise (`jevons-gemma4-diffusion` R5).
- **Whole requests.** The figures in `benchmarks/README.md` hold: Snake prefill p50 688 ms,
  canvas forward p50 57 ms, JevBench p50 0.400 s.

The decision is per component. A hybrid in which only the components with no measured loss move
is a valid outcome.

## Specs it touches

None yet. A port would touch `jevons-burn` (a quantized product as a backend extension, a GGUF
loader path) and `jevons-gemma4-diffusion`. `jevons-kernels` keeps its kernels either way.
