//! Stands in for ROCm's `hipconfig` on build machines without ROCm.
//!
//! `cubecl-hip-sys` loads HIP at run time but picks its binding layout at build time from
//! `hipconfig --version`; without it, it takes the newest layout it ships. Release builds pin
//! the layout of the ROCm version the service is tested with (`JEVONS_HIP_VERSION`).
fn main() {
    let version = option_env!("JEVONS_HIP_VERSION").unwrap_or("7.2.53211");
    match std::env::args().nth(1).as_deref() {
        Some("--version") => println!("{version}-jevons"),
        Some("-R") | Some("-p") => println!("/opt/rocm"),
        _ => println!("{version}-jevons"),
    }
}
