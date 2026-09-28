# Vendored stylo 0.8.0

A copy of [stylo](https://github.com/servo/stylo) 0.8.0 (MPL-2.0), used through
`[patch.crates-io]` in the workspace `Cargo.toml` for dioxus-native / Blitz 0.2, which pins
`stylo = "=0.8.0"`.

The only change is in `lib.rs`: `#[macro_use(Add, AddAssign, Deref, DerefMut, From)] extern crate
derive_more;` instead of a glob `#[macro_use]`. Cargo unifies features across the build, and the
inference stack (cubecl) enables derive_more's `debug` and `eq` features; the glob import then
brings derive_more's `Debug` and `PartialEq` derives into stylo, shadowing the standard ones, and
compilation fails with E0275 (overflow evaluating `Debug`/`PartialEq` for recursive types).

Drop this copy when Blitz moves to a stylo release without the glob import.
