//! Bounded decoding of compressed request images.
use crate::{Error, ImageInput, Result};
use image::{ImageFormat, ImageReader, Limits};
use std::io::Cursor;

/// A decoded image in 8-bit RGB, row-major.
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Decodes JPEG, PNG, WebP or GIF bytes with allocation limits.
pub fn decode_image(input: &ImageInput) -> Result<RgbImage> {
    if input.bytes.is_empty() || input.bytes.len() > 5 * 1024 * 1024 {
        return Err(Error::InvalidInput(
            "Image must contain 1 byte to 5 MiB".into(),
        ));
    }
    let format = image::guess_format(&input.bytes)
        .map_err(|_| Error::InvalidInput("Unknown image format".into()))?;
    if !matches!(
        format,
        ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::WebP | ImageFormat::Gif
    ) {
        return Err(Error::InvalidInput(
            "Supported images: JPEG, PNG, WebP, GIF".into(),
        ));
    }
    let mut reader = ImageReader::with_format(Cursor::new(&input.bytes), format);
    let mut limits = Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode().map_err(|_| {
        Error::InvalidInput(
            "Invalid image or image exceeds decoder limits (8192 pixels per side, 64 MiB)".into(),
        )
    })?;
    if u64::from(image.width()) * u64::from(image.height()) > 16_777_216 {
        return Err(Error::InvalidInput("Image exceeds 16 megapixels".into()));
    }
    let rgb = image.to_rgb8();
    Ok(RgbImage {
        width: rgb.width() as usize,
        height: rgb.height() as usize,
        data: rgb.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_formats_decode_to_rgb_and_truncated_images_fail() {
        for format in [
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::Gif,
            ImageFormat::WebP,
        ] {
            let image = image::RgbImage::from_pixel(2, 3, image::Rgb([10, 20, 30]));
            let mut output = Cursor::new(Vec::new());
            image.write_to(&mut output, format).unwrap();
            let mut input = ImageInput {
                bytes: output.into_inner(),
            };
            let decoded = decode_image(&input).unwrap();
            assert_eq!((decoded.width, decoded.height), (2, 3));
            input.bytes.truncate(8);
            assert!(decode_image(&input).is_err());
        }
    }
}
