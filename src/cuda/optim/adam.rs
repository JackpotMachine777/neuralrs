//! Adam over resident parameters (mirror of CPU `ADAM`).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void adam(float* param, const float* grad, float* m, float* v,
        float lr, float beta1, float beta2, float eps, float bc1, float bc2, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            float m_hat = m[i] / bc1;
            float v_hat = v[i] / bc2;
            param[i] = param[i] - lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Adam. Moment buffers allocated lazily on first step; in-place update.
pub struct ADAM {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
}

impl ADAM {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self { lr, beta1, beta2, epsilon, t: 0, m: Vec::new(), v: Vec::new() }
    }

    /// # Panics
    /// If a parameter is not on the GPU, or the parameter count changes between steps.
    pub fn step(&mut self, params: &[Rc<RefCell<Node>>]) {
        let stream = backend::stream();
        if self.t == 0 {
            for p in params {
                let len: usize = p.borrow().shape.iter().product();
                self.m.push(stream.alloc_zeros::<f32>(len).expect("adam: m alloc failed"));
                self.v.push(stream.alloc_zeros::<f32>(len).expect("adam: v alloc failed"));
            }
        }
        assert_eq!(params.len(), self.m.len(), "adam: parameter count changed between steps");

        let bc1 = 1.0 - self.beta1.powi(self.t as i32 + 1);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32 + 1);
        let (lr, beta1, beta2, eps) = (self.lr, self.beta1, self.beta2, self.epsilon);
        let f = module().load_function("adam").expect("adam not found");

        for (i, p) in params.iter().enumerate() {
            let mut node = p.borrow_mut();
            let len: usize = node.shape.iter().product();
            let gpu = node.gpu.as_mut().expect("adam: param not on GPU");
            let GpuBuffers { data, grad } = gpu;
            let g = grad.borrow();
            let n = len as i32;
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut builder = stream.launch_builder(&f);
            builder.arg(data);
            builder.arg(&*g);
            builder.arg(&mut self.m[i]);
            builder.arg(&mut self.v[i]);
            builder.arg(&lr);
            builder.arg(&beta1);
            builder.arg(&beta2);
            builder.arg(&eps);
            builder.arg(&bc1);
            builder.arg(&bc2);
            builder.arg(&n);
            unsafe { builder.launch(cfg).expect("adam: launch failed"); }
        }
        self.t += 1;
    }
}