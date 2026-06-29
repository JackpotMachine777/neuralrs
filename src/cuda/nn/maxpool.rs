//! 2-D max pooling on the GPU (resident autograd op).
//!
//! Takes the largest value in each `kernel × kernel` window (general `stride`,
//! no padding), mirroring the CPU `MaxPool2d`. The forward records the winning
//! input index per window (argmax); backward scatters the gradient to those
//! winners with atomicAdd (windows can overlap when stride < kernel, so the
//! contributions accumulate, exactly the CPU's `+=`).

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void maxpool_forward(
        float* out, int* argmax, const float* input,
        int N, int C, int in_h, int in_w, int k, int stride, int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * C * out_h * out_w;
        if (idx >= total) return;

        int ox = idx % out_w;
        int oy = (idx / out_w) % out_h;
        int ch = (idx / (out_w * out_h)) % C;
        int ni = idx / (out_w * out_h * C);

        int first = ((ni * C + ch) * in_h + oy * stride) * in_w + ox * stride;
        float best = input[first];
        int best_idx = first;

        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                int iy = oy * stride + i;
                int ix = ox * stride + j;
                int in_idx = ((ni * C + ch) * in_h + iy) * in_w + ix;
                float v = input[in_idx];
                if (v > best) { best = v; best_idx = in_idx; }
            }

        out[idx] = best;
        argmax[idx] = best_idx;
    }

    extern "C" __global__ void maxpool_backward(
        float* dinput, const float* grad, const int* argmax, int total
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;
        atomicAdd(&dinput[argmax[idx]], grad[idx]);
    }
"#;

crate::kernel_module!(KERNEL);

/// 2-D max pooling of a resident input, kept on the GPU. Input
/// `[N, C, in_h, in_w]`, `kernel × kernel` windows with `stride`, no padding;
/// output `[N, C, out_h, out_w]`.
///
/// # Panics
/// If the input is not on the GPU.
pub fn maxpool2d(input: &Rc<RefCell<Node>>, kernel: usize, stride: usize) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, argmax, n, c, in_h, in_w, out_h, out_w) = {
        let in_n = input.borrow();
        let in_gpu = in_n.gpu.as_ref().expect("cuda maxpool: input not on GPU");
        let (n, c, in_h, in_w) = (in_n.shape[0], in_n.shape[1], in_n.shape[2], in_n.shape[3]);
        let out_h = (in_h - kernel) / stride + 1;
        let out_w = (in_w - kernel) / stride + 1;
        let total = n * c * out_h * out_w;

        let mut out = stream.alloc_zeros::<f32>(total).expect("cuda maxpool: out alloc failed");
        let mut argmax = stream.alloc_zeros::<i32>(total).expect("cuda maxpool: argmax alloc failed");

        let f = module().load_function("maxpool_forward").expect("maxpool_forward not found");
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let (ni, ci, ih, iw) = (n as i32, c as i32, in_h as i32, in_w as i32);
        let (k, st, oh, ow) = (kernel as i32, stride as i32, out_h as i32, out_w as i32);
        let mut b = stream.launch_builder(&f);
        b.arg(&mut out); b.arg(&mut argmax); b.arg(&in_gpu.data);
        b.arg(&ni); b.arg(&ci); b.arg(&ih); b.arg(&iw);
        b.arg(&k); b.arg(&st); b.arg(&oh); b.arg(&ow);
        unsafe { b.launch(cfg).expect("cuda maxpool: forward launch failed"); }

        (out, argmax, n, c, in_h, in_w, out_h, out_w)
    };

    let out_node = Node::new(vec![], vec![n, c, out_h, out_w]);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone()];

        let out_len = n * c * out_h * out_w;
        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(out_len).expect("cuda maxpool: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        let in_len = n * c * in_h * in_w;

        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();

            let mut dinput = stream.alloc_zeros::<f32>(in_len).expect("cuda maxpool bwd: alloc failed");

            let f = module().load_function("maxpool_backward").expect("maxpool_backward not found");
            let cfg = LaunchConfig::for_num_elems(out_len as u32);
            let t = out_len as i32;
            let mut b = stream.launch_builder(&f);
            b.arg(&mut dinput);
            b.arg(&*og);
            b.arg(&argmax);
            b.arg(&t);
            unsafe { b.launch(cfg).expect("cuda maxpool bwd: launch failed"); }

            accumulate_into(&in_bwd, &Rc::new(RefCell::new(dinput)), in_len);
        }));
    }

    out_node
}