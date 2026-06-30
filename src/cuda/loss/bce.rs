//! Binary cross-entropy on the GPU (operates on resident predictions).
//!
//! Mirrors the CPU `bce` / `bce_grad`: forward returns
//! -mean(target*ln(pred) + (1-target)*ln(1-pred)); backward seeds the
//! prediction's gradient with (pred-target)/(pred*(1-pred)). Note the gradient
//! is NOT averaged over n, matching the CPU `bce_grad` exactly.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::Node;
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void bce_term(float* out, const float* pred, const float* target, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = target[i] * logf(pred[i]) + (1.0f - target[i]) * logf(1.0f - pred[i]);
    }
    extern "C" __global__ void bce_grad(float* grad, const float* pred, const float* target, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) grad[i] = (pred[i] - target[i]) / (pred[i] * (1.0f - pred[i]));
    }
"#;
crate::kernel_module!(KERNEL);

/// Binary cross-entropy of resident predictions against a resident target,
/// averaged over all elements. Predictions must lie in (0, 1). Both nodes must
/// be on the GPU.
///
/// # Panics
/// If either node is not on the GPU.
pub fn bce(pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32 {
    let stream = backend::stream();
    let p = pred.borrow();
    let t = target.borrow();
    let len: usize = p.shape.iter().product();
    let pd = &p.gpu.as_ref().expect("cuda bce: pred not on GPU").data;
    let td = &t.gpu.as_ref().expect("cuda bce: target not on GPU").data;

    let mut term = stream.alloc_zeros::<f32>(len).expect("cuda bce: alloc failed");
    let f = module().load_function("bce_term").expect("bce_term not found");
    let n = len as i32;
    let cfg = LaunchConfig::for_num_elems(len as u32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut term);
    builder.arg(pd);
    builder.arg(td);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("cuda bce: forward launch failed"); }

    let host = stream.clone_dtoh(&term).expect("cuda bce: dtoh failed");
    -host.iter().sum::<f32>() / len as f32
}

/// Seeds the prediction's gradient with (pred-target)/(pred*(1-pred)), writing
/// into the resident grad buffer (unnormalized, matching the CPU `bce_grad`).
/// Run `backward_graph` afterwards.
///
/// # Panics
/// If either node is not on the GPU.
pub fn bce_backward(pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) {
    let stream = backend::stream();
    let p = pred.borrow();
    let t = target.borrow();
    let len: usize = p.shape.iter().product();
    let p_gpu = p.gpu.as_ref().expect("cuda bce: pred not on GPU");
    let pd = &p_gpu.data;
    let td = &t.gpu.as_ref().expect("cuda bce: target not on GPU").data;
    let mut grad = p_gpu.grad.borrow_mut();

    let f = module().load_function("bce_grad").expect("bce_grad not found");
    let n = len as i32;
    let cfg = LaunchConfig::for_num_elems(len as u32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut *grad);
    builder.arg(pd);
    builder.arg(td);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("cuda bce: backward launch failed"); }
}