//! Reduction primitives over a 2D tensor (device-level, no autograd).
//!
//! Shared building blocks for ops that collapse an axis: the bias gradient sums
//! over the batch, and later softmax / cross-entropy will sum and max over the
//! feature axis. Naive one-thread-per-output kernels for now, tree reduction is
//! a later perf pass.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void sum_axis0(float* out, const float* in, int rows, int cols) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < cols) {
            float acc = 0.0f;
            for (int r = 0; r < rows; ++r) acc += in[r * cols + c];
            out[c] = acc;
        }
    }
"#;

crate::kernel_module!(KERNEL);

/// Sums a `[rows, cols]` device buffer over the outer axis into a `[cols]` buffer
/// (`out[c] = Σ_r in[r, c]`). The bias-gradient pattern: collapse the batch, keep
/// the features.
pub(crate) fn sum_axis0(input: &CudaSlice<f32>, rows: usize, cols: usize) -> CudaSlice<f32> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(cols).expect("sum_axis0: alloc failed");

    let f = module().load_function("sum_axis0").expect("sum_axis0 not found");
    let cfg = LaunchConfig::for_num_elems(cols as u32);
    let (r, c) = (rows as i32, cols as i32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut out);
    builder.arg(input);
    builder.arg(&r);
    builder.arg(&c);
    unsafe { builder.launch(cfg).expect("sum_axis0: launch failed"); }

    out
}