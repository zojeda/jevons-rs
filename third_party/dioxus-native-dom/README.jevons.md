# Vendored dioxus-native-dom 0.7.10

A copy of [dioxus-native-dom](https://github.com/DioxusLabs/dioxus) 0.7.10 (MIT OR Apache-2.0),
used through `[patch.crates-io]` in the workspace `Cargo.toml`.

The only change is in `src/mutation_writer.rs` (marked `jevons:`). When Dioxus reuses an
`ElementId`, `assign_node_id` drops the node that id mapped to if it is unparented, with its
subtree. Other `ElementId`s mapped into that subtree kept pointing at the freed slots; once the
slab reused a slot (for example for a new template prototype, which is unparented by design),
reusing one of those ids dropped that unrelated node, and cloning the template later panicked in
`blitz-dom` (`node_at_path`: "invalid key" or "index out of bounds"). The patch keeps a reverse
map and forgets every mapping into a subtree before dropping it.

Drop this copy when dioxus-native-dom fixes the stale mappings upstream.
