//! Renders the icons to PNG for the documentation, and optionally scales down an artwork file.
//!
//! ```bash
//! cargo run -p jevons-desktop-core --example render_icons -- crates/jevons-desktop/assets [artwork.png]
//! ```
//!
//! Writes `icon-256.png` (the app icon), `tray.png` (every tray state at 32 px, shown at 2×), and
//! with an artwork file, `jevons.png` (the artwork at 512 px).
use image::{ImageBuffer, Rgba, RgbaImage, imageops};
use jevons_desktop_core::icons::{self, SIZE, TrayState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().ok_or("usage: render_icons <dir> [artwork]")?);
    std::fs::create_dir_all(&out)?;

    let icon: RgbaImage =
        ImageBuffer::from_raw(256, 256, icons::app_icon(256)).ok_or("icon size")?;
    icon.save(out.join("icon-256.png"))?;

    // One tile per state the tray can show, in reading order.
    let states = [
        TrayState::Idle,
        TrayState::Tuning,
        TrayState::Offline,
        TrayState::Error,
        TrayState::Listening { level: 4 },
        TrayState::Listening { level: 14 },
        TrayState::Transcribing { frame: 0 },
        TrayState::Thinking { frame: 1 },
    ];
    let frames = icons::frames();
    let (scale, gap) = (2, 12);
    let tile = SIZE * scale;
    let mut sheet = RgbaImage::from_pixel(
        states.len() as u32 * (tile + gap) + gap,
        tile + 2 * gap,
        Rgba([32, 33, 38, 255]),
    );
    for (i, state) in states.iter().enumerate() {
        let (_, pixels) = &frames[icons::frame_index(*state)];
        let frame: RgbaImage =
            ImageBuffer::from_raw(SIZE, SIZE, pixels.clone()).ok_or("frame size")?;
        let big = imageops::resize(&frame, tile, tile, imageops::FilterType::Nearest);
        imageops::overlay(
            &mut sheet,
            &big,
            i64::from(gap + i as u32 * (tile + gap)),
            i64::from(gap),
        );
    }
    sheet.save(out.join("tray.png"))?;

    if let Some(artwork) = args.next() {
        let art = image::open(artwork)?.to_rgba8();
        imageops::resize(&art, 512, 512, imageops::FilterType::Lanczos3)
            .save(out.join("jevons.png"))?;
    }
    Ok(())
}
