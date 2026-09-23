use crate::{Error, ImageInput, Result};
use image::{ImageFormat, ImageReader, Limits, RgbImage};
use std::io::Cursor;

pub(crate) fn decode(input: &ImageInput) -> Result<RgbImage> {
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
    Ok(image.to_rgb8())
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
            let image = RgbImage::from_pixel(2, 3, image::Rgb([10, 20, 30]));
            let mut output = Cursor::new(Vec::new());
            image.write_to(&mut output, format).unwrap();
            let mut input = ImageInput {
                bytes: output.into_inner(),
            };
            assert_eq!(decode(&input).unwrap().dimensions(), (2, 3));
            input.bytes.truncate(8);
            assert!(decode(&input).is_err());
        }
    }
}
