//! Batch normalization for 2-D inputs [batch, features] on the GPU (resident).
//!
//! Normalizes each feature across the batch, then scales/shifts with gamma/beta,
//! mirroring the CPU `BatchNorm` (biased variance, running stats with momentum,
//! train/eval modes). Running stats are caller-owned resident nodes, updated in
//! place during training. Per-feature reductions run in one thread each (the
//! batch is the reduced axis); the backward derives sum1 = gamma*dbeta and
//! sum2 = gamma*dgamma from those reductions.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void bn_stats(
        float* mean, float* inv_std, float* running_mean, float* running_var,
        const float* input, int batch, int features, float eps, float momentum, int training
    ) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= features) return;

        if (training) {
            float m = 0.0f;
            for (int b = 0; b < batch; ++b) m += input[b * features + f];
            m /= (float)batch;

            float v = 0.0f;
            for (int b = 0; b < batch; ++b) {
                float d = input[b * features + f] - m;
                v += d * d;
            }
            v /= (float)batch;

            running_mean[f] = momentum * running_mean[f] + (1.0f - momentum) * m;
            running_var[f]  = momentum * running_var[f]  + (1.0f - momentum) * v;

            mean[f] = m;
            inv_std[f] = 1.0f / sqrtf(v + eps);
        } else {
            mean[f] = running_mean[f];
            inv_std[f] = 1.0f / sqrtf(running_var[f] + eps);
        }
    }

    extern "C" __global__ void bn_normalize(
        float* out, float* x_norm, const float* input,
        const float* mean, const float* inv_std, const float* gamma, const float* beta,
        int batch, int features
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * features;
        if (idx >= total) return;

        int f = idx % features;
        float xn = (input[idx] - mean[f]) * inv_std[f];
        x_norm[idx] = xn;
        out[idx] = gamma[f] * xn + beta[f];
    }

    extern "C" __global__ void bn_reduce(
        float* dgamma, float* dbeta, const float* grad, const float* x_norm,
        int batch, int features
    ) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= features) return;

        float dg = 0.0f, db = 0.0f;
        for (int b = 0; b < batch; ++b) {
            int idx = b * features + f;
            float g = grad[idx];
            dg += g * x_norm[idx];
            db += g;
        }
        dgamma[f] = dg;
        dbeta[f] = db;
    }

    extern "C" __global__ void bn_dx(
        float* dinput, const float* grad, const float* x_norm,
        const float* gamma, const float* inv_std,
        const float* dgamma, const float* dbeta,
        int batch, int features, int training
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * features;
        if (idx >= total) return;

        int f = idx % features;
        float gf = gamma[f];

        if (training) {
            float n = (float)batch;
            float sum1 = gf * dbeta[f];
            float sum2 = gf * dgamma[f];
            float dxhat = grad[idx] * gf;
            dinput[idx] = inv_std[f] * (dxhat - sum1 / n - x_norm[idx] * sum2 / n);
        } else {
            dinput[idx] = grad[idx] * gf * inv_std[f];
        }
    }
"#;

crate::kernel_module!(KERNEL);

/// Batch-normalizes a resident input `[batch, features]` with resident
/// gamma/beta `[features]`, kept on the GPU. `running_mean`/`running_var` are
/// resident nodes (length `features`); in training they're updated in place.
/// `out = gamma * (x - mean) / sqrt(var + eps) + beta`.
///
/// # Panics
/// If any node is not on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm(
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

    let (out_data, x_norm, inv_std, shape, batch, features, total) = {
        let in_n = input.borrow();
        let g_n = gamma.borrow();
        let b_n = beta.borrow();
        let mut rm_n = running_mean.borrow_mut();
        let mut rv_n = running_var.borrow_mut();

        let in_gpu = in_n.gpu.as_ref().expect("cuda bn: input not on GPU");
        let g_gpu = g_n.gpu.as_ref().expect("cuda bn: gamma not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda bn: beta not on GPU");
        let rm = &mut rm_n.gpu.as_mut().expect("cuda bn: running_mean not on GPU").data;
        let rv = &mut rv_n.gpu.as_mut().expect("cuda bn: running_var not on GPU").data;

        let (batch, features) = (in_n.shape[0], in_n.shape[1]);
        let total = batch * features;

        let mut mean = stream.alloc_zeros::<f32>(features).expect("cuda bn: mean alloc failed");
        let mut inv_std = stream.alloc_zeros::<f32>(features).expect("cuda bn: inv_std alloc failed");

        let f = module().load_function("bn_stats").expect("bn_stats not found");
        let cfg = LaunchConfig::for_num_elems(features as u32);
        let (bi, fi, tr) = (batch as i32, features as i32, training as i32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut mean); bd.arg(&mut inv_std); bd.arg(rm); bd.arg(rv);
        bd.arg(&in_gpu.data); bd.arg(&bi); bd.arg(&fi); bd.arg(&eps); bd.arg(&momentum); bd.arg(&tr);
        unsafe { bd.launch(cfg).expect("cuda bn: stats launch failed"); }

        let mut out = stream.alloc_zeros::<f32>(total).expect("cuda bn: out alloc failed");
        let mut x_norm = stream.alloc_zeros::<f32>(total).expect("cuda bn: x_norm alloc failed");
        let f2 = module().load_function("bn_normalize").expect("bn_normalize not found");
        let cfg2 = LaunchConfig::for_num_elems(total as u32);
        let mut bd2 = stream.launch_builder(&f2);
        bd2.arg(&mut out); bd2.arg(&mut x_norm); bd2.arg(&in_gpu.data);
        bd2.arg(&mean); bd2.arg(&inv_std); bd2.arg(&g_gpu.data); bd2.arg(&b_gpu.data);
        bd2.arg(&bi); bd2.arg(&fi);
        unsafe { bd2.launch(cfg2).expect("cuda bn: normalize launch failed"); }

        (out, x_norm, inv_std, in_n.shape.clone(), batch, features, total)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone(), gamma.clone(), beta.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(total).expect("cuda bn: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        let g_bwd = gamma.clone();
        let b_bwd = beta.clone();

        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();
            let (bi, fi, tr) = (batch as i32, features as i32, training as i32);

            let mut dgamma = stream.alloc_zeros::<f32>(features).expect("cuda bn bwd: dgamma alloc");
            let mut dbeta = stream.alloc_zeros::<f32>(features).expect("cuda bn bwd: dbeta alloc");
            let fr = module().load_function("bn_reduce").expect("bn_reduce not found");
            let cfg_f = LaunchConfig::for_num_elems(features as u32);
            let mut b1 = stream.launch_builder(&fr);
            b1.arg(&mut dgamma); b1.arg(&mut dbeta); b1.arg(&*og); b1.arg(&x_norm); b1.arg(&bi); b1.arg(&fi);
            unsafe { b1.launch(cfg_f).expect("cuda bn bwd: reduce launch failed"); }

            let dinput = {
                let g_n = g_bwd.borrow();
                let g_data = &g_n.gpu.as_ref().expect("cuda bn bwd: gamma not on GPU").data;
                let mut dinput = stream.alloc_zeros::<f32>(total).expect("cuda bn bwd: dinput alloc");
                let fx = module().load_function("bn_dx").expect("bn_dx not found");
                let cfg_t = LaunchConfig::for_num_elems(total as u32);
                let mut b2 = stream.launch_builder(&fx);
                b2.arg(&mut dinput); b2.arg(&*og); b2.arg(&x_norm); b2.arg(g_data); b2.arg(&inv_std);
                b2.arg(&dgamma); b2.arg(&dbeta); b2.arg(&bi); b2.arg(&fi); b2.arg(&tr);
                unsafe { b2.launch(cfg_t).expect("cuda bn bwd: dx launch failed"); }
                dinput
            };

            accumulate_into(&g_bwd, &Rc::new(RefCell::new(dgamma)), features);
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(dbeta)), features);
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(dinput)), total);
        }));
    }

    out_node
}