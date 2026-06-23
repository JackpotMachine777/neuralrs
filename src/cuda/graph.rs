//! GPU-resident autograd graph ops.
//!
//! Unlike the device-level helpers in `add`/`matmul` (which copy host -> device
//! -> host on every call), these run on data that already lives on the GPU and
//! leave the result there. Chain several together and the intermediates never
//! touch the CPU, that's the point of residency, and what makes training fast
//! instead of drowning in PCIe transfers.
//!
//! Move a node onto the device with [`to_cuda`], build a graph with the resident
//! ops, then read a result back with [`to_host`].

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaModule, LaunchConfig, PushKernelArg, CudaSlice};

use super::backend;
use crate::autograd::node::{GpuBuffers, Node};

const KERNEL: &str = r#"
    extern "C" __global__ void accumulate(float* acc, const float* src, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) acc[i] += src[i];
    }
"#;

static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();

fn module() -> &'static Arc<CudaModule> {
    MODULE.get_or_init(|| backend::compile(KERNEL))
}

/// Moves a node's data onto the GPU (host -> device) and allocates a zeroed
/// gradient buffer there too. Afterwards the node's `gpu` field is populated.
pub fn to_cuda(node: &Rc<RefCell<Node>>) {
    let stream = backend::stream();
    let mut n = node.borrow_mut();

    let data = stream.clone_htod(&n.data).expect("to_cuda: htod failed");
    let len = n.data.len();
    let grad = stream.alloc_zeros::<f32>(len).expect("to_cuda: device alloc failed");

    n.gpu = Some(GpuBuffers {
        data,
        grad: Rc::new(RefCell::new(grad)),
    });
}

/// Reads a resident node's data back to host memory (device -> host).
///
/// # Panics
/// If the node is not currently on the GPU.
pub fn to_host(node: &Rc<RefCell<Node>>) -> Vec<f32> {
    let stream = backend::stream();
    let n = node.borrow();
    let gpu = n.gpu.as_ref().expect("to_host: node is not on the GPU");

    stream.clone_dtoh(&gpu.data).expect("to_host: dtoh failed")
}

/// Launches `target.grad += src` on the GPU. The accumulation pattern at the
/// heart of autograd, gradients add up, they don't overwrite.
pub(crate) fn accumulate_into(target: &Rc<RefCell<Node>>, src: &Rc<RefCell<CudaSlice<f32>>>, n: usize) {
    let stream = backend::stream();
    let t = target.borrow();
    let t_gpu = t.gpu.as_ref().expect("backward: parent not on GPU");
    let mut dst = t_gpu.grad.borrow_mut();
    let s = src.borrow();

    let acc = module().load_function("accumulate").expect("accumulate not found");
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut builder = stream.launch_builder(&acc);
    builder.arg(&mut *dst);
    builder.arg(&*s);
    builder.arg(&n);
    unsafe {
        builder.launch(cfg).expect("accumulate launch failed");
    }
}

/// Overwrites a resident node's gradient with the given values (host -> device).
/// Used to seed the upstream gradient before a backward pass.
pub fn set_grad(node: &Rc<RefCell<Node>>, grad: &[f32]) {
    let stream = backend::stream();
    let n = node.borrow();
    let gpu = n.gpu.as_ref().expect("set_grad: node not on GPU");
    let new = stream.clone_htod(grad).expect("set_grad: htod failed");
    *gpu.grad.borrow_mut() = new;
}

/// Reads a resident node's gradient back to host memory (device -> host).
pub fn read_grad(node: &Rc<RefCell<Node>>) -> Vec<f32> {
    let stream = backend::stream();
    let n = node.borrow();
    let gpu = n.gpu.as_ref().expect("read_grad: node not on GPU");
    let g = gpu.grad.borrow();

    stream.clone_dtoh(&*g).expect("read_grad: dtoh failed")
}