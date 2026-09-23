//! Image preprocessing for the Gemma 4 vision encoder, matching llama.cpp's `mtmd`:
//! aspect-preserving target size aligned to `patch * merge`, Pillow-compatible fixed-point
//! bicubic resampling, centered black padding, and patch extraction scaled to `[-1, 1]`.

/// Token budget and geometry of the encoder input.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub patch: usize,
    pub merge: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl Geometry {
    /// Resized image size: dimensions are multiples of `patch * merge` and the number of
    /// output tokens `(w / align) * (h / align)` lies within the budget where possible.
    pub fn target_size(&self, width: usize, height: usize) -> (usize, usize) {
        let align = self.patch * self.merge;
        let area = align * align;
        let (min_pixels, max_pixels) = (self.min_tokens * area, self.max_tokens * area);
        let f = align as f32;
        let round = |x: f32| (x / f).round() as usize * align;
        let ceil = |x: f32| (x / f).ceil() as usize * align;
        let floor = |x: f32| (x / f).floor() as usize * align;
        let (wf, hf) = (width as f32, height as f32);
        let mut w = align.max(round(wf));
        let mut h = align.max(round(hf));
        if h * w > max_pixels {
            let beta = (hf * wf / max_pixels as f32).sqrt();
            h = align.max(floor(hf / beta));
            w = align.max(floor(wf / beta));
        } else if h * w < min_pixels {
            let beta = (min_pixels as f32 / (hf * wf)).sqrt();
            h = ceil(hf * beta);
            w = ceil(wf * beta);
        }
        (w, h)
    }
}

/// Packed RGB8 image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Resizes to `(width, height)` preserving the aspect ratio (scale rounded up, capped) and
/// centers the result on black, as llama.cpp's default `PAD_CEIL` resize.
pub fn resize_padded(src: &Rgb, width: usize, height: usize) -> Rgb {
    if (src.width, src.height) == (width, height) {
        return src.clone();
    }
    let scale = (width as f32 / src.width as f32).min(height as f32 / src.height as f32);
    let nw = ((src.width as f32 * scale).ceil() as usize).min(width);
    let nh = ((src.height as f32 * scale).ceil() as usize).min(height);
    let resized = resize_bicubic(src, nw, nh);
    let mut out = Rgb {
        width,
        height,
        data: vec![0; width * height * 3],
    };
    let (ox, oy) = ((width - nw) / 2, (height - nh) / 2);
    for y in 0..nh {
        let d = ((y + oy) * width + ox) * 3;
        out.data[d..d + nw * 3].copy_from_slice(&resized.data[y * nw * 3..(y + 1) * nw * 3]);
    }
    out
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Per output pixel: first input index, count and fixed-point weights (Pillow `precompute_coeffs`).
fn coefficients(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = vec![0i32; out_size * ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let pre: Vec<f64> = (0..xmax)
            .map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss))
            .collect();
        let total: f64 = pre.iter().sum();
        for (x, w) in pre.iter().enumerate() {
            let w = if total != 0.0 { w / total } else { *w };
            let fixed = w * f64::from(1u32 << PRECISION_BITS);
            weights[xx * ksize + x] = (fixed + if w < 0.0 { -0.5 } else { 0.5 }) as i32;
        }
        bounds.push((xmin, xmax));
    }
    (ksize, bounds, weights)
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Separable Pillow bicubic resample (horizontal pass, then vertical).
pub fn resize_bicubic(src: &Rgb, width: usize, height: usize) -> Rgb {
    let mut cur = src.clone();
    if width != cur.width {
        let (ksize, bounds, k) = coefficients(cur.width, width);
        let mut data = vec![0u8; width * cur.height * 3];
        for y in 0..cur.height {
            for (xx, &(xmin, count)) in bounds.iter().enumerate() {
                for ch in 0..3 {
                    let mut acc = 1i32 << (PRECISION_BITS - 1);
                    for x in 0..count {
                        acc += i32::from(cur.data[(y * cur.width + xmin + x) * 3 + ch])
                            * k[xx * ksize + x];
                    }
                    data[(y * width + xx) * 3 + ch] = clip8(acc);
                }
            }
        }
        cur = Rgb {
            width,
            height: cur.height,
            data,
        };
    }
    if height != cur.height {
        let (ksize, bounds, k) = coefficients(cur.height, height);
        let row = cur.width * 3;
        let mut data = vec![0u8; row * height];
        for (yy, &(ymin, count)) in bounds.iter().enumerate() {
            for i in 0..row {
                let mut acc = 1i32 << (PRECISION_BITS - 1);
                for y in 0..count {
                    acc += i32::from(cur.data[(ymin + y) * row + i]) * k[yy * ksize + y];
                }
                data[yy * row + i] = clip8(acc);
            }
        }
        cur = Rgb {
            width: cur.width,
            height,
            data,
        };
    }
    cur
}

/// Patch rows `[(h/p) * (w/p), 3 * p * p]` in raster order, each ordered channel, row, column,
/// with pixel values mapped to `2 * v / 255 - 1`.
pub fn patches(img: &Rgb, patch: usize) -> Vec<f32> {
    let (cols, rows) = (img.width / patch, img.height / patch);
    let len = 3 * patch * patch;
    let mut out = vec![0.0f32; rows * cols * len];
    for py in 0..rows {
        for px in 0..cols {
            let base = (py * cols + px) * len;
            for ch in 0..3 {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let (x, y) = (px * patch + kx, py * patch + ky);
                        let v = f32::from(img.data[(y * img.width + x) * 3 + ch]) / 255.0;
                        out[base + (ch * patch + ky) * patch + kx] = v * 2.0 - 1.0;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMMA4: Geometry = Geometry {
        patch: 16,
        merge: 3,
        min_tokens: 70,
        max_tokens: 280,
    };

    fn tokens((w, h): (usize, usize)) -> usize {
        (w / 48) * (h / 48)
    }

    #[test]
    fn target_sizes_match_llama_cpp_token_counts() {
        // Token counts reported by llama.cpp's mtmd for the same inputs.
        assert_eq!(GEMMA4.target_size(330, 220), (528, 336));
        assert_eq!(tokens(GEMMA4.target_size(330, 220)), 77);
        assert_eq!(tokens(GEMMA4.target_size(224, 224)), 81);
        assert_eq!(tokens(GEMMA4.target_size(640, 360)), 104);
        assert_eq!(tokens(GEMMA4.target_size(300, 700)), 90);
        assert_eq!(tokens(GEMMA4.target_size(64, 48)), 80);
        let big = GEMMA4.target_size(4000, 3000);
        assert!(tokens(big) <= 280 && big.0.is_multiple_of(48) && big.1.is_multiple_of(48));
    }

    #[test]
    fn bicubic_resize_preserves_flat_colors_and_pads_with_black() {
        let src = Rgb {
            width: 30,
            height: 20,
            data: [200u8, 100, 50].repeat(600),
        };
        let out = resize_padded(&src, 96, 48);
        // Scale min(96/30, 48/20) = 2.4 -> 72 x 48, centered with 12 black columns each side.
        assert_eq!(&out.data[..3], &[0, 0, 0]);
        let mid = (24 * 96 + 48) * 3;
        assert_eq!(&out.data[mid..mid + 3], &[200, 100, 50]);
        let up = resize_bicubic(&src, 17, 43);
        assert!(up.data.chunks(3).all(|p| p == [200, 100, 50]));
    }

    #[test]
    fn patches_are_channel_major_and_scaled_to_unit_range() {
        let mut img = Rgb {
            width: 32,
            height: 16,
            data: vec![0; 32 * 16 * 3],
        };
        // Second patch, green channel, row 1, column 2.
        img.data[(32 + 18) * 3 + 1] = 255;
        let p = patches(&img, 16);
        assert_eq!(p.len(), 2 * 768);
        assert_eq!(p[768 + 256 + 16 + 2], 1.0);
        assert_eq!(p[0], -1.0);
    }
}
