//! Adagrad over resident parameters (mirror of CPU `Adagrad`).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void adagrad(float* param, const float* grad, float* g_sum,
        float lr, float eps, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            float g = grad[i];
            g_sum[i] = g_sum[i] + g * g;
            param[i] = param[i] - lr * g / (sqrtf(g_sum[i]) + eps);
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Adagrad. Accumulated squared-gradient buffer allocated lazily on first step.
pub struct Adagrad {
    pub lr: f32,
    pub epsilon: f32,
    g_sum: Vec<CudaSlice<f32>>,
}

impl Adagrad {
    pub fn new(lr: f32, epsilon: f32) -> Self {
        Self { lr, epsilon, g_sum: Vec::new() }
    }

    /// # Panics
    /// If a parameter is not on the GPU, or the parameter count changes between steps.
    pub fn step(&mut self, params: &[Rc<RefCell<Node>>]) {
        let stream = backend::stream();
        if self.g_sum.is_empty() {
            for p in params {
                let len: usize = p.borrow().shape.iter().product();
                self.g_sum.push(stream.alloc_zeros::<f32>(len).expect("adagrad: g_sum alloc failed"));
            }
        }
        assert_eq!(params.len(), self.g_sum.len(), "adagrad: parameter count changed between steps");

        let (lr, eps) = (self.lr, self.epsilon);
        let f = module().load_function("adagrad").expect("adagrad not found");

        for (i, p) in params.iter().enumerate() {
            let mut node = p.borrow_mut();
            let len: usize = node.shape.iter().product();
            let gpu = node.gpu.as_mut().expect("adagrad: param not on GPU");
            let GpuBuffers { data, grad } = gpu;
            let g = grad.borrow();
            let n = len as i32;
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut builder = stream.launch_builder(&f);
            builder.arg(data);
            builder.arg(&*g);
            builder.arg(&mut self.g_sum[i]);
            builder.arg(&lr);
            builder.arg(&eps);
            builder.arg(&n);
            unsafe { builder.launch(cfg).expect("adagrad: launch failed"); }
        }
    }
}