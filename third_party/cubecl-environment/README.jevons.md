# Vendored cubecl-environment 0.11.0-pre.4

A copy of [cubecl-environment](https://github.com/tracel-ai/cubecl) (MIT OR Apache-2.0), used
through `[patch.crates-io]` in the workspace `Cargo.toml`. The only change is in
`src/persistence/turso.rs`: on Windows the cache database opens without Turso's experimental
multi-process WAL.

Upstream asks for that WAL on every platform. Turso's Windows I/O backend does not have it, so
the writable open failed ("experimental multiprocess WAL is not supported by the active IO
backend"), the read-only fallback found an empty file ("no such table: meta"), and CubeCL kept
its compiled kernels and tuned plans in memory only. Every process then compiled and tuned
again: minutes at each start of the tray app (issue #17).

Without that WAL one process writes the cache at a time. A second process that finds it locked
caches nothing for that run, as upstream already handles.

Drop this copy once upstream opens the cache on Windows.
