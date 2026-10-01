# Vendored blitz-paint 0.2.1

A copy of [blitz-paint](https://github.com/DioxusLabs/blitz) 0.2.1 (MIT OR Apache-2.0), used through
`[patch.crates-io]` in the workspace `Cargo.toml` for dioxus-native 0.7.10.

The only change is in `src/render.rs`, `draw_text_input_text` (marked `jevons:`): the text caret
is painted in the field's text colour, as CSS's `caret-color: auto` asks. The original painted it
black, which the inspector's dark theme made invisible in every text field.

Drop this copy when Blitz paints the caret in `caret-color` upstream.
