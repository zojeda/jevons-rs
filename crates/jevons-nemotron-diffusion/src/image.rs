//! Image preprocessing for the Pixtral vision tower, matching the checkpoint's
//! `image_processing.py` (a port of `mistral_common`'s image encoder): the longest edge is capped
//! at 1400 pixels, the size is rounded up to 28-pixel cells, the image is resized with OpenCV's
//! `INTER_CUBIC`, and pixels are normalized with the CLIP mean and standard deviation.
use jevons_core::RgbImage;

pub const PATCH: usize = 14;
pub const MERGE: usize = 2;
const CELL: usize = PATCH * MERGE;
const MAX_EDGE: f64 = 1400.0;
// Values as written in the reference (`image_processing.py`).
#[allow(clippy::excessive_precision)]
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Merged-token grid of an image: `(width, height)` in 28-pixel cells.
pub fn token_grid(width: usize, height: usize) -> (usize, usize) {
    let (mut w, mut h) = (width as f64, height as f64);
    let ratio = (h / MAX_EDGE).max(w / MAX_EDGE);
    if ratio > 1.0 {
        // Python's round(): halves go to the even neighbor.
        w = (w / ratio).round_ties_even();
        h = (h / ratio).round_ties_even();
    }
    let (w, h) = (w as usize, h as usize);
    ((w.max(1) - 1) / CELL + 1, (h.max(1) - 1) / CELL + 1)
}

/// OpenCV's bicubic weights (`A = -0.75`) for fractional offset `x` in `[0, 1)`.
fn cubic_weights(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let w0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let w1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let w2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [w0, w1, w2, 1.0 - w0 - w1 - w2]
}

/// Source taps and weights along one axis, as `cv::resize` computes them: centers aligned,
/// no antialiasing, border pixels replicated.
fn taps(src: usize, dst: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let f = (d as f64 + 0.5) * scale - 0.5;
            let base = f.floor();
            let weights = cubic_weights((f - base) as f32);
            let index = |k: i64| (base as i64 + k).clamp(0, src as i64 - 1) as usize;
            ([index(-1), index(0), index(1), index(2)], weights)
        })
        .collect()
}

/// Resizes to `(width, height)` with OpenCV `INTER_CUBIC` on float RGB in `[0, 255]`, returning
/// row-major interleaved RGB.
pub fn resize_cubic(image: &RgbImage, width: usize, height: usize) -> Vec<f32> {
    let (sw, sh) = (image.width, image.height);
    let xs = taps(sw, width);
    let ys = taps(sh, height);
    // Horizontal pass into [sh, width, 3], then vertical.
    let mut rows = vec![0f32; sh * width * 3];
    for y in 0..sh {
        let src = &image.data[y * sw * 3..(y + 1) * sw * 3];
        for (x, (idx, w)) in xs.iter().enumerate() {
            for c in 0..3 {
                rows[(y * width + x) * 3 + c] =
                    (0..4).map(|k| w[k] * f32::from(src[idx[k] * 3 + c])).sum();
            }
        }
    }
    let mut out = vec![0f32; height * width * 3];
    for (y, (idx, w)) in ys.iter().enumerate() {
        for i in 0..width * 3 {
            out[y * width * 3 + i] = (0..4).map(|k| w[k] * rows[idx[k] * width * 3 + i]).sum();
        }
    }
    out
}

/// A preprocessed image: patches in raster order, each flattened as `(channel, row, column)`
/// like the patch convolution's weights.
pub struct Patches {
    /// `[rows * cols, 3 * 14 * 14]` normalized values.
    pub data: Vec<f32>,
    /// Patch grid (rows, columns).
    pub rows: usize,
    pub cols: usize,
    /// Merged-token grid (width, height).
    pub tokens: (usize, usize),
}

/// Normalized `[3, height, width]` planes of the resized image (the reference's pixel array).
pub fn normalized_planes(image: &RgbImage) -> (Vec<f32>, usize, usize) {
    let (tw, th) = token_grid(image.width, image.height);
    let (width, height) = (tw * CELL, th * CELL);
    let resized = resize_cubic(image, width, height);
    let mut planes = vec![0f32; 3 * height * width];
    for i in 0..height * width {
        for c in 0..3 {
            planes[c * height * width + i] = (resized[i * 3 + c] / 255.0 - MEAN[c]) / STD[c];
        }
    }
    (planes, width, height)
}

pub fn patches(image: &RgbImage) -> Patches {
    let (planes, width, height) = normalized_planes(image);
    let (rows, cols) = (height / PATCH, width / PATCH);
    let size = 3 * PATCH * PATCH;
    let mut data = vec![0f32; rows * cols * size];
    for r in 0..rows {
        for q in 0..cols {
            let patch = &mut data[(r * cols + q) * size..(r * cols + q + 1) * size];
            for c in 0..3 {
                for ky in 0..PATCH {
                    for kx in 0..PATCH {
                        patch[(c * PATCH + ky) * PATCH + kx] =
                            planes[c * height * width + (r * PATCH + ky) * width + q * PATCH + kx];
                    }
                }
            }
        }
    }
    Patches {
        data,
        rows,
        cols,
        tokens: (cols / MERGE, rows / MERGE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_grids_cap_the_longest_edge_and_round_up_to_cells() {
        assert_eq!(token_grid(300, 200), (11, 8));
        assert_eq!(token_grid(28, 28), (1, 1));
        assert_eq!(token_grid(29, 1), (2, 1));
        // 2800x1400 halves to 1400x700: 50 x 25 cells.
        assert_eq!(token_grid(2800, 1400), (50, 25));
        // 4201x1 has ratio 3.0007; width rounds to 1400.
        assert_eq!(token_grid(4201, 3), (50, 1));
    }

    #[test]
    fn cubic_resize_keeps_constants_and_matches_opencv_weights() {
        let image = RgbImage {
            width: 3,
            height: 2,
            data: vec![10; 3 * 2 * 3],
        };
        assert!(
            resize_cubic(&image, 7, 5)
                .iter()
                .all(|v| (v - 10.0).abs() < 1e-4)
        );
        // Weights sum to one; the center tap dominates at offset 0.
        let w = cubic_weights(0.0);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(w[1], 1.0);
        let w = cubic_weights(0.5);
        assert!((w[0] - -0.09375).abs() < 1e-6 && (w[1] - 0.59375).abs() < 1e-6);
    }

    #[test]
    fn patches_follow_the_convolution_weight_layout() {
        let image = RgbImage {
            width: 28,
            height: 28,
            data: (0..28 * 28 * 3).map(|i| (i % 251) as u8).collect(),
        };
        let p = patches(&image);
        assert_eq!((p.rows, p.cols, p.tokens), (2, 2, (1, 1)));
        let (planes, width, height) = normalized_planes(&image);
        // Patch (1, 0), channel 2, row 3, column 5.
        let got = p.data[(2 * 3 * 196) + (2 * 14 + 3) * 14 + 5];
        let want = planes[2 * height * width + (14 + 3) * width + 5];
        assert_eq!(got, want);
    }
}
