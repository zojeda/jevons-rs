# Vendored cubecl-hip-sys 7.14.6085001

A copy of [cubecl-hip-sys](https://github.com/tracel-ai/cubecl-hip) (MIT OR Apache-2.0),
used through `[patch.crates-io]` in the workspace `Cargo.toml`. The only change is in
`src/dynamic.rs`, so Windows finds the HIP libraries it loads at run time:

- the driver's versioned runtime `amdhip64_7.dll` / `amdhip64_6.dll` before `amdhip64.dll`,
  which can be an older runtime without the ROCm 6/7 entry points;
- the HIP SDK's versioned `hiprtcXXYY.dll`, in `HIP_PATH\bin`;
- libraries bundled next to the executable.

Drop this copy once upstream loads these names.
