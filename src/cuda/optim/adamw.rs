//! AdamW over resident parameters, Adam with decoupled weight decay.
//!
//! First/second moment buffers (`m`, `v`) live on the device, one pair per
//! parameter, and the parameters update in place, nothing round-trips to host
//! during training. Decoupled weight decay (the "W"): the decay is applied to the
//! weights directly, not folded into the gradient.
//! State can be exported and re-imported for checkpointing, so training
//! resumes with the moments intact (see `export_state`/`import_state`).

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void adamw(
        float* param, const float* grad, float* m, float* v,
        float lr, float beta1, float beta2, float eps, float weight_decay,
        float bc1, float bc2, int n
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

            float m_hat = m[i] / bc1;
            float v_hat = v[i] / bc2;

            param[i] = param[i] - lr * m_hat / (sqrtf(v_hat) + eps) - lr * weight_decay * param[i];
        }
    }
"#;

crate::kernel_module!(KERNEL);

/// AdamW over resident parameters. Moment buffers are allocated lazily on the
/// first [`step`](Self::step), matched to the parameters passed; parameters
/// update in place and their gradients must already be on the GPU.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub t: usize,
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
}

impl AdamW {
    /// AdamW with the given hyperparameters.
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self { lr, beta1, beta2, epsilon, weight_decay, t: 0, m: Vec::new(), v: Vec::new() }
    }

    /// One in-place AdamW update over the given resident parameters.
    ///
    /// # Panics
    /// If any parameter is not on the GPU, or the parameter count changes between
    /// steps (moment buffers are bound to the first step's parameters).
    pub fn step(&mut self, params: &[Rc<RefCell<Node>>]) {
        let stream = backend::stream();

        if self.m.is_empty() {
            for p in params {
                let len: usize = p.borrow().shape.iter().product();
                self.m.push(stream.alloc_zeros::<f32>(len).expect("adamw: m alloc failed"));
                self.v.push(stream.alloc_zeros::<f32>(len).expect("adamw: v alloc failed"));
            }
        }
        assert_eq!(params.len(), self.m.len(), "adamw: parameter count changed between steps");

        let bc1 = 1.0 - self.beta1.powi(self.t as i32 + 1);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32 + 1);
        let (lr, beta1, beta2, eps, wd) =
            (self.lr, self.beta1, self.beta2, self.epsilon, self.weight_decay);

        let f = module().load_function("adamw").expect("adamw not found");

        for (i, p) in params.iter().enumerate() {
            let mut node = p.borrow_mut();
            let len: usize = node.shape.iter().product();
            let gpu = node.gpu.as_mut().expect("adamw: param not on GPU");
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
            builder.arg(&wd);
            builder.arg(&bc1);
            builder.arg(&bc2);
            builder.arg(&n);
            unsafe { builder.launch(cfg).expect("adamw: launch failed"); }
        }

        self.t += 1;
    }

    /// Downloads the optimizer state, the step counter and every moment
    /// buffer, so it can be checkpointed alongside the parameters. The
    /// tensors come back as all first moments in parameter order, then all
    /// second moments. Empty if the optimizer hasn't stepped yet.
    pub fn export_state(&self) -> (usize, Vec<Vec<f32>>) {
        let stream = backend::stream();
        let mut out = Vec::with_capacity(self.m.len() * 2);
        for s in self.m.iter().chain(self.v.iter()) {
            out.push(stream.clone_dtoh(s).expect("adamw: state dtoh failed"));
        }
        (self.t, out)
    }

    /// Restores state produced by [`export_state`](Self::export_state):
    /// `moments` is every first moment in parameter order followed by every
    /// second moment (so the count must be even), `t` the step counter. Call
    /// before the first [`step`](Self::step); the buffers must match the
    /// parameters that will be passed there.
    pub fn import_state(&mut self, t: usize, moments: &[Vec<f32>]) {
        assert!(moments.len() % 2 == 0, "adamw: moments must be all m's then all v's");
        let stream = backend::stream();
        let half = moments.len() / 2;
        self.m = moments[..half]
            .iter()
            .map(|h| stream.clone_htod(h).expect("adamw: state htod failed"))
            .collect();
        self.v = moments[half..]
            .iter()
            .map(|h| stream.clone_htod(h).expect("adamw: state htod failed"))
            .collect();
        self.t = t;
    }
}