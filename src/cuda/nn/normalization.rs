//! Layer normalization on the GPU (resident). Normalizes each row over its last
//! dim (features), then scales/shifts with resident gamma/beta. Mirrors the CPU
//! `LayerNorm`: same in train and eval, no running stats. Per-row reductions run
//! one thread each (features is the reduced axis); param grads reduce over rows.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void ln_forward(
        float* out, float* x_norm, float* inv_std,
        const float* input, const float* gamma, const float* beta,
        int rows, int features, float eps
    ) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) return;
        int start = r * features;

        float mean = 0.0f;
        for (int f = 0; f < features; ++f) mean += input[start + f];
        mean /= (float)features;

        float var = 0.0f;
        for (int f = 0; f < features; ++f) { float d = input[start + f] - mean; var += d * d; }
        var /= (float)features;

        float istd = 1.0f / sqrtf(var + eps);
        inv_std[r] = istd;

        for (int f = 0; f < features; ++f) {
            float xn = (input[start + f] - mean) * istd;
            x_norm[start + f] = xn;
            out[start + f] = gamma[f] * xn + beta[f];
        }
    }

    extern "C" __global__ void ln_reduce(
        float* dgamma, float* dbeta, const float* grad, const float* x_norm,
        int rows, int features
    ) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= features) return;
        float dg = 0.0f, db = 0.0f;
        for (int r = 0; r < rows; ++r) {
            int idx = r * features + f;
            dg += grad[idx] * x_norm[idx];
            db += grad[idx];
        }
        dgamma[f] = dg;
        dbeta[f] = db;
    }

    extern "C" __global__ void ln_dx(
        float* dinput, const float* grad, const float* gamma,
        const float* x_norm, const float* inv_std, int rows, int features
    ) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) return;
        int start = r * features;
        float istd = inv_std[r];

        float sum1 = 0.0f, sum2 = 0.0f;
        for (int f = 0; f < features; ++f) {
            float dxhat = grad[start + f] * gamma[f];
            sum1 += dxhat;
            sum2 += dxhat * x_norm[start + f];
        }

        float nf = (float)features;
        for (int f = 0; f < features; ++f) {
            float dxhat = grad[start + f] * gamma[f];
            dinput[start + f] = istd * (dxhat - sum1 / nf - x_norm[start + f] * sum2 / nf);
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Layer-normalizes a resident input over its last dim, with resident gamma/beta
/// `[features]`. `out = gamma * (x - mean) / sqrt(var + eps) + beta`, per row.
///
/// # Panics
/// If any node is not on the GPU.
pub fn layernorm(
    input: &Rc<RefCell<Node>>,
    gamma: &Rc<RefCell<Node>>,
    beta: &Rc<RefCell<Node>>,
    eps: f32,
) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, x_norm, inv_std, shape, rows, features, total) = {
        let in_n = input.borrow();
        let g_n = gamma.borrow();
        let b_n = beta.borrow();
        let in_gpu = in_n.gpu.as_ref().expect("cuda layernorm: input not on GPU");
        let g_gpu = g_n.gpu.as_ref().expect("cuda layernorm: gamma not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda layernorm: beta not on GPU");

        let features = *in_n.shape.last().unwrap();
        let total: usize = in_n.shape.iter().product();
        let rows = total / features;

        let mut out = stream.alloc_zeros::<f32>(total).expect("cuda layernorm: out alloc");
        let mut x_norm = stream.alloc_zeros::<f32>(total).expect("cuda layernorm: x_norm alloc");
        let mut inv_std = stream.alloc_zeros::<f32>(rows).expect("cuda layernorm: inv_std alloc");

        let f = module().load_function("ln_forward").expect("ln_forward not found");
        let cfg = LaunchConfig::for_num_elems(rows as u32);
        let (rr, ff) = (rows as i32, features as i32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&mut x_norm); bd.arg(&mut inv_std);
        bd.arg(&in_gpu.data); bd.arg(&g_gpu.data); bd.arg(&b_gpu.data);
        bd.arg(&rr); bd.arg(&ff); bd.arg(&eps);
        unsafe { bd.launch(cfg).expect("cuda layernorm: forward launch failed"); }

        (out, x_norm, inv_std, in_n.shape.clone(), rows, features, total)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone(), gamma.clone(), beta.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(total).expect("cuda layernorm: grad alloc")));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        let g_bwd = gamma.clone();
        let b_bwd = beta.clone();
        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();
            let (rr, ff) = (rows as i32, features as i32);

            let mut dgamma = stream.alloc_zeros::<f32>(features).expect("cuda layernorm bwd: dgamma alloc");
            let mut dbeta = stream.alloc_zeros::<f32>(features).expect("cuda layernorm bwd: dbeta alloc");
            let fr = module().load_function("ln_reduce").expect("ln_reduce not found");
            let cfg_f = LaunchConfig::for_num_elems(features as u32);
            let mut b1 = stream.launch_builder(&fr);
            b1.arg(&mut dgamma); b1.arg(&mut dbeta); b1.arg(&*og); b1.arg(&x_norm); b1.arg(&rr); b1.arg(&ff);
            unsafe { b1.launch(cfg_f).expect("cuda layernorm bwd: reduce launch failed"); }

            let dinput = {
                let g_n = g_bwd.borrow();
                let g_data = &g_n.gpu.as_ref().expect("cuda layernorm bwd: gamma not on GPU").data;
                let mut dinput = stream.alloc_zeros::<f32>(total).expect("cuda layernorm bwd: dinput alloc");
                let fx = module().load_function("ln_dx").expect("ln_dx not found");
                let cfg_r = LaunchConfig::for_num_elems(rows as u32);
                let mut b2 = stream.launch_builder(&fx);
                b2.arg(&mut dinput); b2.arg(&*og); b2.arg(g_data); b2.arg(&x_norm); b2.arg(&inv_std); b2.arg(&rr); b2.arg(&ff);
                unsafe { b2.launch(cfg_r).expect("cuda layernorm bwd: dx launch failed"); }
                dinput
            };

            accumulate_into(&g_bwd, &Rc::new(RefCell::new(dgamma)), features);
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(dbeta)), features);
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(dinput)), total);
        }));
    }
    out_node
}