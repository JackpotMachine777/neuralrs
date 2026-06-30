//! 2-D average pooling on the GPU (resident autograd op). Mean of each
//! kernel x kernel window (general stride, no padding), mirroring the CPU
//! `AvgPool2d`. Backward spreads each output's gradient evenly over its window
//! with atomicAdd (windows can overlap when stride < kernel).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void avgpool_forward(
        float* out, const float* input,
        int N, int C, int in_h, int in_w, int k, int stride, int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * C * out_h * out_w;
        if (idx >= total) return;

        int ox = idx % out_w;
        int oy = (idx / out_w) % out_h;
        int ch = (idx / (out_w * out_h)) % C;
        int ni = idx / (out_w * out_h * C);

        float area = (float)(k * k);
        float sum = 0.0f;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                int iy = oy * stride + i;
                int ix = ox * stride + j;
                sum += input[((ni * C + ch) * in_h + iy) * in_w + ix];
            }
        out[idx] = sum / area;
    }

    extern "C" __global__ void avgpool_backward(
        float* dinput, const float* grad,
        int N, int C, int in_h, int in_w, int k, int stride, int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * C * out_h * out_w;
        if (idx >= total) return;

        int ox = idx % out_w;
        int oy = (idx / out_w) % out_h;
        int ch = (idx / (out_w * out_h)) % C;
        int ni = idx / (out_w * out_h * C);

        float area = (float)(k * k);
        float g = grad[idx] / area;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                int iy = oy * stride + i;
                int ix = ox * stride + j;
                atomicAdd(&dinput[((ni * C + ch) * in_h + iy) * in_w + ix], g);
            }
    }
"#;
crate::kernel_module!(KERNEL);

/// 2-D average pooling of a resident input, kept on the GPU. Input
/// `[N, C, in_h, in_w]`, `kernel x kernel` windows with `stride`, no padding;
/// output `[N, C, out_h, out_w]`.
///
/// # Panics
/// If the input is not on the GPU.
pub fn avgpool2d(input: &Rc<RefCell<Node>>, kernel: usize, stride: usize) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, n, c, in_h, in_w, out_h, out_w) = {
        let in_n = input.borrow();
        let in_gpu = in_n.gpu.as_ref().expect("cuda avgpool: input not on GPU");
        let (n, c, in_h, in_w) = (in_n.shape[0], in_n.shape[1], in_n.shape[2], in_n.shape[3]);
        let out_h = (in_h - kernel) / stride + 1;
        let out_w = (in_w - kernel) / stride + 1;
        let total = n * c * out_h * out_w;

        let mut out = stream.alloc_zeros::<f32>(total).expect("cuda avgpool: out alloc failed");
        let f = module().load_function("avgpool_forward").expect("avgpool_forward not found");
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let (ni, ci, ih, iw) = (n as i32, c as i32, in_h as i32, in_w as i32);
        let (k, st, oh, ow) = (kernel as i32, stride as i32, out_h as i32, out_w as i32);
        let mut b = stream.launch_builder(&f);
        b.arg(&mut out); b.arg(&in_gpu.data);
        b.arg(&ni); b.arg(&ci); b.arg(&ih); b.arg(&iw);
        b.arg(&k); b.arg(&st); b.arg(&oh); b.arg(&ow);
        unsafe { b.launch(cfg).expect("cuda avgpool: forward launch failed"); }

        (out, n, c, in_h, in_w, out_h, out_w)
    };

    let out_node = Node::new(vec![], vec![n, c, out_h, out_w]);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone()];
        let out_len = n * c * out_h * out_w;
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(out_len).expect("cuda avgpool: grad alloc")));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        let in_len = n * c * in_h * in_w;
        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();
            let mut dinput = stream.alloc_zeros::<f32>(in_len).expect("cuda avgpool bwd: alloc failed");
            let f = module().load_function("avgpool_backward").expect("avgpool_backward not found");
            let cfg = LaunchConfig::for_num_elems(out_len as u32);
            let (ni, ci, ih, iw) = (n as i32, c as i32, in_h as i32, in_w as i32);
            let (k, st, oh, ow) = (kernel as i32, stride as i32, out_h as i32, out_w as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut dinput); b.arg(&*og);
            b.arg(&ni); b.arg(&ci); b.arg(&ih); b.arg(&iw);
            b.arg(&k); b.arg(&st); b.arg(&oh); b.arg(&ow);
            unsafe { b.launch(cfg).expect("cuda avgpool bwd: launch failed"); }
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(dinput)), in_len);
        }));
    }
    out_node
}