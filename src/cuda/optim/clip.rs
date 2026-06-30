//! Gradient-norm clipping over resident parameters (mirror of CPU `clip_grad_norm`).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::Node;
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void sumsq(const float* g, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) atomicAdd(out, g[i] * g[i]);
    }
    extern "C" __global__ void clip_scale(float* g, float s, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) g[i] = g[i] * s;
    }
"#;
crate::kernel_module!(KERNEL);

/// Clips the combined L2 gradient norm of the given resident parameters to
/// `max_norm`. If the total norm across all parameter gradients exceeds it,
/// every gradient is scaled by `max_norm / total_norm`.
///
/// # Panics
/// If any parameter is not on the GPU.
pub fn clip_grad_norm(params: &[Rc<RefCell<Node>>], max_norm: f32) {
    let stream = backend::stream();

    let mut sumsq_dev = stream.alloc_zeros::<f32>(1).expect("clip: sumsq alloc failed");
    let f_sumsq = module().load_function("sumsq").expect("sumsq not found");

    for p in params {
        let node = p.borrow();
        let len: usize = node.shape.iter().product();
        let gpu = node.gpu.as_ref().expect("clip: param not on GPU");
        let g = gpu.grad.borrow();
        let n = len as i32;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut builder = stream.launch_builder(&f_sumsq);
        builder.arg(&*g);
        builder.arg(&mut sumsq_dev);
        builder.arg(&n);
        unsafe { builder.launch(cfg).expect("clip: sumsq launch failed"); }
    }

    let total_norm = stream.clone_dtoh(&sumsq_dev).expect("clip: dtoh failed")[0].sqrt();

    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        let f_scale = module().load_function("clip_scale").expect("clip_scale not found");
        for p in params {
            let node = p.borrow();
            let len: usize = node.shape.iter().product();
            let gpu = node.gpu.as_ref().expect("clip: param not on GPU");
            let mut g = gpu.grad.borrow_mut();
            let n = len as i32;
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut builder = stream.launch_builder(&f_scale);
            builder.arg(&mut *g);
            builder.arg(&scale);
            builder.arg(&n);
            unsafe { builder.launch(cfg).expect("clip: scale launch failed"); }
        }
    }
}