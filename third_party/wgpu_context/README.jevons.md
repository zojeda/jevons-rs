# Vendored wgpu_context 0.1.2

A copy of [wgpu_context](https://github.com/dioxuslabs/anyrender) 0.1.2 (MIT OR Apache-2.0), used
through `[patch.crates-io]` in the workspace `Cargo.toml` for anyrender_vello 0.6.2, which renders
the jevons-desktop windows.

The changes are marked `jevons:`.

`src/surface_renderer.rs`, `maybe_blit_and_present`: a frame the surface cannot take is skipped.
The original called `expect` on `Surface::get_current_texture`, on the application's main thread:
any surface error ended jevons-desktop with the models still loaded, which has left HIP failing
with status 719 for every process. The first failure is logged, and so is the first frame presented
after it. A surface that is `Lost` or `Outdated` is configured again. A lost device, which wgpu
reports as `Other`, does not come back: the window shows nothing more until the application starts
again, and the application goes on without it.

`src/lib.rs`, `create_device`: a device-lost callback logs the reason and the driver's message.
wgpu reports a lost device to that callback alone.

`Cargo.toml` adds `log` for both.

Not changed: `current_surface_texture` and the branch of `target_texture_view` without an
intermediate texture still panic on a surface error. Vello renders into the intermediate texture,
so neither runs here. anyrender_vello's own `expect` on `render_to_texture` and `unwrap` on
`Device::poll` are outside this crate.

Drop this copy when anyrender handles surface errors upstream.
