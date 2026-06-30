//! SGD with momentum over resident parameters (mirror of CPU `SGD`).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void sgd(float* param, const float* grad, float* vel,
        float lr, float momentum, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            vel[i] = momentum * vel[i] + grad[i];
            param[i] = param[i] - lr * vel[i];
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// SGD with momentum. Velocity buffers are allocated lazily on the first step;
/// parameters update in place and their gradients must already be on the GPU.
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    velocity: Vec<CudaSlice<f32>>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self { lr, momentum, velocity: Vec::new() }
    }

    /// # Panics
    /// If a parameter is not on the GPU, or the parameter count changes between steps.
    pub fn step(&mut self, params: &[Rc<RefCell<Node>>]) {
        let stream = backend::stream();
        if self.velocity.is_empty() {
            for p in params {
                let len: usize = p.borrow().shape.iter().product();
                self.velocity.push(stream.alloc_zeros::<f32>(len).expect("sgd: velocity alloc failed"));
            }
        }
        assert_eq!(params.len(), self.velocity.len(), "sgd: parameter count changed between steps");

        let (lr, momentum) = (self.lr, self.momentum);
        let f = module().load_function("sgd").expect("sgd not found");

        for (i, p) in params.iter().enumerate() {
            let mut node = p.borrow_mut();
            let len: usize = node.shape.iter().product();
            let gpu = node.gpu.as_mut().expect("sgd: param not on GPU");
            let GpuBuffers { data, grad } = gpu;
            let g = grad.borrow();
            let n = len as i32;
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut builder = stream.launch_builder(&f);
            builder.arg(data);
            builder.arg(&*g);
            builder.arg(&mut self.velocity[i]);
            builder.arg(&lr);
            builder.arg(&momentum);
            builder.arg(&n);
            unsafe { builder.launch(cfg).expect("sgd: launch failed"); }
        }
    }
}