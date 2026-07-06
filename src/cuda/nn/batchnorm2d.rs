//! Batch normalization for 4-D conv inputs [batch, channels, height, width]
//! on the GPU (resident).
//!
//! Like the resident `batchnorm`, but normalizes per channel, pooling the
//! statistics over the batch and both spatial dimensions, mirroring the CPU
//! `BatchNorm2d` (biased variance, running stats with momentum, train/eval
//! modes). Running stats are caller-owned resident nodes, updated in place
//! during training. Per-channel reductions run one thread per channel over
//! batch*h*w elements, fine at parity scale, a candidate for block
//! reductions if profiling ever shows them hot.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void bn2d_stats(
        float* mean, float* inv_std, float* running_mean, float* running_var,
        const float* input, int batch, int channels, int hw,
        float eps, float momentum, int training
    ) {
        int ch = blockIdx.x * blockDim.x + threadIdx.x;
        if (ch >= channels) return;

        if (training) {
            float count = (float)(batch * hw);

            float m = 0.0f;
            for (int b = 0; b < batch; ++b) {
                int base = (b * channels + ch) * hw;
                for (int p = 0; p < hw; ++p) m += input[base + p];
            }
            m /= count;

            float v = 0.0f;
            for (int b = 0; b < batch; ++b) {
                int base = (b * channels + ch) * hw;
                for (int p = 0; p < hw; ++p) {
                    float d = input[base + p] - m;
                    v += d * d;
                }
            }
            v /= count;

            running_mean[ch] = momentum * running_mean[ch] + (1.0f - momentum) * m;
            running_var[ch]  = momentum * running_var[ch]  + (1.0f - momentum) * v;

            mean[ch] = m;
            inv_std[ch] = 1.0f / sqrtf(v + eps);
        } else {
            mean[ch] = running_mean[ch];
            inv_std[ch] = 1.0f / sqrtf(running_var[ch] + eps);
        }
    }

    extern "C" __global__ void bn2d_normalize(
        float* out, float* x_norm, const float* input,
        const float* mean, const float* inv_std, const float* gamma, const float* beta,
        int batch, int channels, int hw
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * channels * hw;
        if (idx >= total) return;

        int ch = (idx / hw) % channels;
        float xn = (input[idx] - mean[ch]) * inv_std[ch];
        x_norm[idx] = xn;
        out[idx] = gamma[ch] * xn + beta[ch];
    }

    extern "C" __global__ void bn2d_reduce(
        float* dgamma, float* dbeta, const float* grad, const float* x_norm,
        int batch, int channels, int hw
    ) {
        int ch = blockIdx.x * blockDim.x + threadIdx.x;
        if (ch >= channels) return;

        float dg = 0.0f, db = 0.0f;
        for (int b = 0; b < batch; ++b) {
            int base = (b * channels + ch) * hw;
            for (int p = 0; p < hw; ++p) {
                float g = grad[base + p];
                dg += g * x_norm[base + p];
                db += g;
            }
        }
        dgamma[ch] = dg;
        dbeta[ch] = db;
    }

    extern "C" __global__ void bn2d_dx(
        float* dinput, const float* grad, const float* x_norm,
        const float* gamma, const float* inv_std,
        const float* dgamma, const float* dbeta,
        int batch, int channels, int hw, int training
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * channels * hw;
        if (idx >= total) return;

        int ch = (idx / hw) % channels;
        float gc = gamma[ch];

        if (training) {
            float count = (float)(batch * hw);
            float sum1 = gc * dbeta[ch];
            float sum2 = gc * dgamma[ch];
            float dxhat = grad[idx] * gc;
            dinput[idx] = inv_std[ch] * (dxhat - sum1 / count - x_norm[idx] * sum2 / count);
        } else {
            dinput[idx] = grad[idx] * gc * inv_std[ch];
        }
    }
"#;

crate::kernel_module!(KERNEL);

/// Batch-normalizes a resident 4-D input `[batch, channels, h, w]` with
/// resident gamma/beta `[channels]`, kept on the GPU. Statistics pool over the
/// batch and both spatial dimensions. `running_mean`/`running_var` are
/// resident nodes (length `channels`); in training they're updated in place.
///
/// # Panics
/// If any node is not on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm2d(
    input: &Rc<RefCell<Node>>,
    gamma: &Rc<RefCell<Node>>,
    beta: &Rc<RefCell<Node>>,
    running_mean: &Rc<RefCell<Node>>,
    running_var: &Rc<RefCell<Node>>,
    momentum: f32,
    eps: f32,
    training: bool,
) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, x_norm, inv_std, shape, batch, channels, hw, total) = {
        let in_n = input.borrow();
        let g_n = gamma.borrow();
        let b_n = beta.borrow();
        let mut rm_n = running_mean.borrow_mut();
        let mut rv_n = running_var.borrow_mut();

        let in_gpu = in_n.gpu.as_ref().expect("cuda bn2d: input not on GPU");
        let g_gpu = g_n.gpu.as_ref().expect("cuda bn2d: gamma not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda bn2d: beta not on GPU");
        let rm = &mut rm_n.gpu.as_mut().expect("cuda bn2d: running_mean not on GPU").data;
        let rv = &mut rv_n.gpu.as_mut().expect("cuda bn2d: running_var not on GPU").data;

        let (batch, channels) = (in_n.shape[0], in_n.shape[1]);
        let hw = in_n.shape[2] * in_n.shape[3];
        let total = batch * channels * hw;

        let mut mean = stream.alloc_zeros::<f32>(channels).expect("cuda bn2d: mean alloc failed");
        let mut inv_std = stream.alloc_zeros::<f32>(channels).expect("cuda bn2d: inv_std alloc failed");

        let f = module().load_function("bn2d_stats").expect("bn2d_stats not found");
        let cfg = LaunchConfig::for_num_elems(channels as u32);
        let (bi, ci, hwi, tr) = (batch as i32, channels as i32, hw as i32, training as i32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut mean); bd.arg(&mut inv_std); bd.arg(rm); bd.arg(rv);
        bd.arg(&in_gpu.data); bd.arg(&bi); bd.arg(&ci); bd.arg(&hwi);
        bd.arg(&eps); bd.arg(&momentum); bd.arg(&tr);
        unsafe { bd.launch(cfg).expect("cuda bn2d: stats launch failed"); }

        let mut out = stream.alloc_zeros::<f32>(total).expect("cuda bn2d: out alloc failed");
        let mut x_norm = stream.alloc_zeros::<f32>(total).expect("cuda bn2d: x_norm alloc failed");
        let f2 = module().load_function("bn2d_normalize").expect("bn2d_normalize not found");
        let cfg2 = LaunchConfig::for_num_elems(total as u32);
        let mut bd2 = stream.launch_builder(&f2);
        bd2.arg(&mut out); bd2.arg(&mut x_norm); bd2.arg(&in_gpu.data);
        bd2.arg(&mean); bd2.arg(&inv_std); bd2.arg(&g_gpu.data); bd2.arg(&b_gpu.data);
        bd2.arg(&bi); bd2.arg(&ci); bd2.arg(&hwi);
        unsafe { bd2.launch(cfg2).expect("cuda bn2d: normalize launch failed"); }

        (out, x_norm, inv_std, in_n.shape.clone(), batch, channels, hw, total)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone(), gamma.clone(), beta.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(total).expect("cuda bn2d: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        let g_bwd = gamma.clone();
        let b_bwd = beta.clone();

        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();
            let (bi, ci, hwi, tr) = (batch as i32, channels as i32, hw as i32, training as i32);

            let mut dgamma = stream.alloc_zeros::<f32>(channels).expect("cuda bn2d bwd: dgamma alloc");
            let mut dbeta = stream.alloc_zeros::<f32>(channels).expect("cuda bn2d bwd: dbeta alloc");
            let fr = module().load_function("bn2d_reduce").expect("bn2d_reduce not found");
            let cfg_c = LaunchConfig::for_num_elems(channels as u32);
            let mut b1 = stream.launch_builder(&fr);
            b1.arg(&mut dgamma); b1.arg(&mut dbeta); b1.arg(&*og); b1.arg(&x_norm);
            b1.arg(&bi); b1.arg(&ci); b1.arg(&hwi);
            unsafe { b1.launch(cfg_c).expect("cuda bn2d bwd: reduce launch failed"); }

            let dinput = {
                let g_n = g_bwd.borrow();
                let g_data = &g_n.gpu.as_ref().expect("cuda bn2d bwd: gamma not on GPU").data;
                let mut dinput = stream.alloc_zeros::<f32>(total).expect("cuda bn2d bwd: dinput alloc");
                let fx = module().load_function("bn2d_dx").expect("bn2d_dx not found");
                let cfg_t = LaunchConfig::for_num_elems(total as u32);
                let mut b2 = stream.launch_builder(&fx);
                b2.arg(&mut dinput); b2.arg(&*og); b2.arg(&x_norm); b2.arg(g_data); b2.arg(&inv_std);
                b2.arg(&dgamma); b2.arg(&dbeta); b2.arg(&bi); b2.arg(&ci); b2.arg(&hwi); b2.arg(&tr);
                unsafe { b2.launch(cfg_t).expect("cuda bn2d bwd: dx launch failed"); }
                dinput
            };

            accumulate_into(&g_bwd, &Rc::new(RefCell::new(dgamma)), channels);
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(dbeta)), channels);
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(dinput)), total);
        }));
    }

    out_node
}