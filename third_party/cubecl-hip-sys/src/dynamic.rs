use libloading::{Library, Symbol};
use std::{env, path::PathBuf, sync::OnceLock};

struct Libraries {
    hip: Library,
    hiprtc: Library,
}

static LIBRARIES: OnceLock<Result<Libraries, String>> = OnceLock::new();

/// Returns whether both HIP runtime libraries can be loaded.
pub fn is_available() -> bool {
    libraries().is_ok()
}

/// Resolve a HIP symbol without making the HIP libraries link-time dependencies.
///
/// This function is called by the generated bindings. A missing runtime is a
/// runtime error because the public binding functions cannot return one common
/// error type for all of their C signatures.
pub(crate) unsafe fn load<T: Copy>(name: &[u8]) -> T {
    let libraries = libraries()
        .as_ref()
        .unwrap_or_else(|error| panic!("{error}"));
    let library = if name.starts_with(b"hiprtc") {
        &libraries.hiprtc
    } else {
        &libraries.hip
    };

    let symbol: Symbol<'_, T> = unsafe { library.get(name) }.unwrap_or_else(|error| {
        let symbol_name = name.strip_suffix(&[0]).unwrap_or(name);
        let symbol = String::from_utf8_lossy(symbol_name);
        panic!("HIP symbol `{symbol}` is unavailable: {error}");
    });
    *symbol
}

fn libraries() -> &'static Result<Libraries, String> {
    LIBRARIES.get_or_init(|| unsafe { load_libraries() })
}

unsafe fn load_libraries() -> Result<Libraries, String> {
    let search_paths = search_paths();
    let hip = load_library("amdhip64", &search_paths)?;
    let hiprtc = load_library("hiprtc", &search_paths)?;
    Ok(Libraries { hip, hiprtc })
}

fn search_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = ["ROCM_PATH", "HIP_PATH"]
        .into_iter()
        .filter_map(env::var_os)
        .flat_map(|path| {
            let path = PathBuf::from(path);
            // The Windows HIP SDK keeps its DLLs in `bin`.
            [path.join("lib"), path.join("bin"), path]
        })
        .collect();
    // Libraries bundled next to the executable.
    if let Some(dir) = env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from)) {
        paths.push(dir);
    }
    paths
}

unsafe fn load_library(name: &str, search_paths: &[PathBuf]) -> Result<Library, String> {
    let mut names = library_names(name);
    if cfg!(target_os = "windows") {
        names.extend(versioned_windows_names(name, search_paths));
    }
    let mut errors = Vec::new();

    for path in search_paths {
        for library_name in &names {
            let candidate = path.join(library_name);
            match unsafe { Library::new(&candidate) } {
                Ok(library) => return Ok(library),
                Err(error) => errors.push(format!("{}: {error}", candidate.display())),
            }
        }
    }

    for library_name in &names {
        match unsafe { Library::new(library_name) } {
            Ok(library) => return Ok(library),
            Err(error) => errors.push(format!("{library_name}: {error}")),
        }
    }

    Err(format!(
        "Could not load HIP library `{name}`. Install ROCm or set ROCM_PATH/HIP_PATH.\n{}",
        errors.join("\n")
    ))
}

fn library_names(name: &str) -> Vec<String> {
    if cfg!(target_os = "windows") {
        // The driver installs the HIP runtime as `amdhip64_<major>.dll`; plain `amdhip64.dll`
        // can be an older runtime without the ROCm 6/7 entry points these bindings use.
        let mut names = Vec::new();
        if name == "amdhip64" {
            names.extend(["amdhip64_7.dll".to_string(), "amdhip64_6.dll".to_string()]);
        }
        names.push(format!("{name}.dll"));
        names
    } else if cfg!(target_os = "macos") {
        vec![format!("lib{name}.dylib")]
    } else {
        vec![
            format!("lib{name}.so"),
            format!("lib{name}.so.1"),
            format!("lib{name}.so.0"),
        ]
    }
}

/// Versioned Windows DLLs such as `hiprtc0702.dll` found in `search_paths`, newest first. The
/// HIP SDK names hiprtc after its major and minor version; its `*-builtins*` companion is loaded
/// by hiprtc itself.
fn versioned_windows_names(name: &str, search_paths: &[PathBuf]) -> Vec<String> {
    let mut found: Vec<String> = search_paths
        .iter()
        .filter_map(|dir| std::fs::read_dir(dir).ok())
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|file| {
            let lower = file.to_ascii_lowercase();
            lower.starts_with(name)
                && lower.ends_with(".dll")
                && !lower.contains("builtins")
                && lower[name.len()..lower.len() - 4]
                    .trim_start_matches('_')
                    .bytes()
                    .all(|b| b.is_ascii_digit())
        })
        .collect();
    found.sort_by(|a, b| b.cmp(a));
    found.dedup();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versioned_dlls_are_found_newest_first_without_builtins() {
        let dir = env::temp_dir().join(format!("hip-names-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in [
            "hiprtc0602.dll",
            "hiprtc0702.dll",
            "hiprtc-builtins0702.dll",
            "hiprtcx.dll",
            "amdhip64_7.dll",
        ] {
            std::fs::write(dir.join(file), b"").unwrap();
        }
        let names = versioned_windows_names("hiprtc", std::slice::from_ref(&dir));
        assert_eq!(names, ["hiprtc0702.dll", "hiprtc0602.dll"]);
        assert_eq!(versioned_windows_names("amdhip64", &[dir.clone()]), ["amdhip64_7.dll"]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
