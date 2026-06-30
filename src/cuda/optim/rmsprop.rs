//! RMSProp over resident parameters (mirror of CPU `RMSProp`).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void rmsprop(float* param, const float* grad, float* v,
        float lr, float beta, float eps, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            float g = grad[i];
            v[i] = beta * v[i] + (1.0f - beta) * g * g;
            param[i] = param[i] - lr * g / (sqrtf(v[i]) + eps);
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// RMSProp. Squared-gradient running average allocated lazily on first step.
pub struct RMSProp {
    pub lr: f32,
    pub beta: f32,
    pub epsilon: f32,
    v: Vec<CudaSlice<f32>>,
}

impl RMSProp {
    pub fn new(lr: f32, beta: f32, epsilon: f32) -> Self {
        Self { lr, beta, epsilon, v: Vec::new() }
    }

    /// # Panics
    /// If a parameter is not on the GPU, or the parameter count changes between steps.
    pub fn step(&mut self, params: &[Rc<RefCell<Node>>]) {
        let stream = backend::stream();
        if self.v.is_empty() {
            for p in params {
                let len: usize = p.borrow().shape.iter().product();
                self.v.push(stream.alloc_zeros::<f32>(len).expect("rmsprop: v alloc failed"));
            }
        }
        assert_eq!(params.len(), self.v.len(), "rmsprop: parameter count changed between steps");

        let (lr, beta, eps) = (self.lr, self.beta, self.epsilon);
        let f = module().load_function("rmsprop").expect("rmsprop not found");

        for (i, p) in params.iter().enumerate() {
            let mut node = p.borrow_mut();
            let len: usize = node.shape.iter().product();
            let gpu = node.gpu.as_mut().expect("rmsprop: param not on GPU");
            let GpuBuffers { data, grad } = gpu;
            let g = grad.borrow();
            let n = len as i32;
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut builder = stream.launch_builder(&f);
            builder.arg(data);
            builder.arg(&*g);
            builder.arg(&mut self.v[i]);
            builder.arg(&lr);
            builder.arg(&beta);
            builder.arg(&eps);
            builder.arg(&n);
            unsafe { builder.launch(cfg).expect("rmsprop: launch failed"); }
        }
    }
}