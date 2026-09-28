# Vendored blitz-dom 0.2.4

A copy of [blitz-dom](https://github.com/DioxusLabs/blitz) 0.2.4 (MIT OR Apache-2.0), used through
`[patch.crates-io]` in the workspace `Cargo.toml` for dioxus-native 0.7.10.

The only change is in `src/mutator.rs`, `add_children_to_parent` (marked `jevons:`): children are
detached from their old parents before they are inserted. The original removed them afterwards,
with `retain`, which also removed a child moved within the same parent from its new position (a
keyed list reordering, such as the profile resolution list when the focused application changes).
The child kept `parent` pointing at a parent that no longer listed it, and the next
`insert_nodes_before` it anchored panicked (`mutator.rs:411`, `unwrap()` on `None`).

`src/lib.rs` also allows `unused_assignments`, a lint newer rustc versions raise on this release.

Drop this copy when Blitz fixes the reparenting upstream.
