//! Positioned reads that leave the file cursor alone, on Unix and Windows.
use std::fs::File;
use std::io;

/// Fills `buf` from `file` at byte `offset`, like Unix `read_exact_at`.
pub(crate) fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match read_at(file, buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill the whole buffer",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positioned_reads_fill_the_buffer_and_report_a_short_file() {
        let path = std::env::temp_dir().join(format!("jevons-io-{}", std::process::id()));
        std::fs::write(&path, b"0123456789").unwrap();
        let file = File::open(&path).unwrap();
        let mut buf = [0u8; 4];
        read_exact_at(&file, &mut buf, 3).unwrap();
        assert_eq!(&buf, b"3456");
        let error = read_exact_at(&file, &mut buf, 8).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        std::fs::remove_file(path).unwrap();
    }
}
