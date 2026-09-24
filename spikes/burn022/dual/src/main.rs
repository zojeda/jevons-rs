#![forbid(unsafe_code)]
//! Item 8: cubecl 0.10 and cubecl 0.11-pre.4 linked into one binary, both initializing HIP.
mod k010 {
    use cubecl010 as cubecl;
    use cubecl::prelude::*;
    #[cube(launch)]
    pub fn add_one(x: &mut Array<f32>) {
        let i = ABSOLUTE_POS as usize;
        if i < x.len() {
            x[i] += 1.0f32;
        }
    }
}
mod k011 {
    use cubecl011 as cubecl;
    use cubecl::prelude::*;
    #[cube(launch)]
    pub fn add_two(x: &mut Tensor<f32>) {
        let i = ABSOLUTE_POS as usize;
        if i < x.len() {
            x[i] += 2.0f32;
        }
    }
}

fn run010() {
    use cubecl010::hip::{AmdDevice, HipRuntime};
    use cubecl010::prelude::*;
    let client = <HipRuntime as Runtime>::client(&AmdDevice::new(0));
    let h = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let arg = TensorBinding::<HipRuntime> {
        handle: h.clone().binding(),
        strides: [1].into(),
        shape: [4].into(),
        runtime: std::marker::PhantomData,
    }
    .into_array_arg();
    k010::add_one::launch::<HipRuntime>(&client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32), arg);
    let out = client.read_one(h).unwrap();
    println!("cubecl 0.10 HIP ({}): {:?}", client.properties().hardware.plane_size_max, f32::from_bytes(&out));
}

fn run011() {
    use cubecl011::prelude::*;
    let client = cubecl011::Device::Hip(cubecl011::hip::AmdDevice::new(0)).client();
    let h = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let arg = TensorBinding {
        handle: h.clone().binding(),
        strides: [1].into(),
        shape: [4].into(),
        tiling: Default::default(),
    }
    .into_tensor_arg();
    k011::add_two::launch(&client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32), arg);
    let out = client.read_one(h).unwrap();
    println!("cubecl 0.11 HIP ({}): {:?}", client.name(), f32::from_bytes(&out));
}

fn main() {
    let order = std::env::args().nth(1).unwrap_or_default();
    if order == "rev" {
        run011();
        run010();
    } else {
        run010();
        run011();
    }
    // and again, interleaved, with both clients alive in-process
    run010();
    run011();
}
