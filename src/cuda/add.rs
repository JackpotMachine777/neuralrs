//! Element-wise tensor add on the GPU: one thread per output element.

use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaModule, LaunchConfig, PushKernelArg};

use super::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void vadd(float* out, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) { out[i] = a[i] + b[i]; }
    }
"#;

/// This op's compiled kernel module, built once on first call.
static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();

fn module() -> &'static Arc<CudaModule> {
    MODULE.get_or_init(|| backend::compile(KERNEL))
}

/// Element-wise sum of two equal-length slices, computed on the GPU.
///
/// Copies both inputs to the device, runs `vadd`, copies the result back.
///
/// # Panics
/// If the two slices have different lengths.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "cuda::add: length mismatch");

    let stream = backend::stream();
    let n = a.len();

    let a_dev = stream.clone_htod(a).expect("htod failed (a)");
    let b_dev = stream.clone_htod(b).expect("htod failed (b)");
    let mut out = stream.alloc_zeros::<f32>(n).expect("device alloc failed");

    let vadd = module().load_function("vadd").expect("kernel 'vadd' not found");
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let mut builder = stream.launch_builder(&vadd);
    builder.arg(&mut out);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("kernel launch failed"); }

    stream.clone_dtoh(&out).expect("dtoh failed")
}