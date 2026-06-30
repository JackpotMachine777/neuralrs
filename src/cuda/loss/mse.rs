//! Mean squared error on the GPU (operates on resident predictions).
//!
//! Mirrors the CPU `MSELoss`: forward returns mean((pred - target)^2) over all
//! elements; backward seeds the prediction's gradient with 2*(pred-target)/n.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::Node;
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void mse_sqdiff(float* out, const float* pred, const float* target, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) { float d = pred[i] - target[i]; out[i] = d * d; }
    }
    extern "C" __global__ void mse_grad(float* grad, const float* pred, const float* target, float inv_n, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) grad[i] = 2.0f * (pred[i] - target[i]) * inv_n;
    }
"#;
crate::kernel_module!(KERNEL);

/// MSE of resident predictions against a resident target, averaged over all
/// elements. Both nodes must be on the GPU.
///
/// # Panics
/// If either node is not on the GPU.
pub fn mse(pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32 {
    let stream = backend::stream();
    let p = pred.borrow();
    let t = target.borrow();
    let len: usize = p.shape.iter().product();
    let pd = &p.gpu.as_ref().expect("cuda mse: pred not on GPU").data;
    let td = &t.gpu.as_ref().expect("cuda mse: target not on GPU").data;

    let mut sq = stream.alloc_zeros::<f32>(len).expect("cuda mse: alloc failed");
    let f = module().load_function("mse_sqdiff").expect("mse_sqdiff not found");
    let n = len as i32;
    let cfg = LaunchConfig::for_num_elems(len as u32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut sq);
    builder.arg(pd);
    builder.arg(td);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("cuda mse: forward launch failed"); }

    let host = stream.clone_dtoh(&sq).expect("cuda mse: dtoh failed");
    host.iter().sum::<f32>() / len as f32
}

/// Seeds the prediction's gradient with 2*(pred-target)/n, writing into the
/// resident grad buffer. Run `backward_graph` afterwards.
///
/// # Panics
/// If either node is not on the GPU.
pub fn mse_backward(pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) {
    let stream = backend::stream();
    let p = pred.borrow();
    let t = target.borrow();
    let len: usize = p.shape.iter().product();
    let p_gpu = p.gpu.as_ref().expect("cuda mse: pred not on GPU");
    let pd = &p_gpu.data;
    let td = &t.gpu.as_ref().expect("cuda mse: target not on GPU").data;
    let mut grad = p_gpu.grad.borrow_mut();

    let inv_n = 1.0f32 / len as f32;
    let f = module().load_function("mse_grad").expect("mse_grad not found");
    let n = len as i32;
    let cfg = LaunchConfig::for_num_elems(len as u32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut *grad);
    builder.arg(pd);
    builder.arg(td);
    builder.arg(&inv_n);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("cuda mse: backward launch failed"); }
}