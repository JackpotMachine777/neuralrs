//! 2-D convolution on the GPU (resident autograd op).
//!
//! Forward is im2col + one GEMM: input patches are unrolled into a
//! [c_in*kh*kw, N*out_h*out_w] matrix, so the whole batch is convolved by a
//! single register-blocked matrix multiply against the [c_out, c_in*kh*kw]
//! weights; a small epilogue kernel then reorders to NCHW and adds the bias.
//! Backward is still direct gather kernels: d_input gathers over (oc,i,j),
//! d_weight runs one thread per (weight, sample) accumulating with atomicAdd,
//! d_bias sums the gradient over (ni,oy,ox).

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;
use crate::cuda::graph::matmul::mm;

const KERNEL: &str = r#"
    // im2col: unrolls input patches into a [c_in*kh*kw, N*out_h*out_w] matrix,
    // one thread per element. Columns are grouped by sample: column index is
    // ni*(out_h*out_w) + oy*out_w + ox. Patch rows follow the weight layout
    // ((ic*kh + i)*kw + j), so the [c_out, c_in*kh*kw] weight matrix multiplies
    // the columns directly, no reshape needed.
    extern "C" __global__ void conv2d_im2col(
        float* cols, const float* input,
        int N, int c_in, int in_h, int in_w,
        int kh, int kw, int stride, int pad,
        int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int col_w = out_h * out_w;
        int width = N * col_w;
        int col_h = c_in * kh * kw;
        int total = col_h * width;
        if (idx >= total) return;

        int row  = idx / width;
        int rest = idx % width;
        int ni = rest / col_w;
        int p  = rest % col_w;
        int oy = p / out_w;
        int ox = p % out_w;

        int j  = row % kw;
        int i  = (row / kw) % kh;
        int ic = row / (kw * kh);

        int iy = oy * stride + i - pad;
        int ix = ox * stride + j - pad;

        float v = 0.0f;
        if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
            v = input[((ni * c_in + ic) * in_h + iy) * in_w + ix];
        cols[idx] = v;
    }

    // Epilogue after the GEMM: reorders the [c_out, N*out_h*out_w] product
    // (columns grouped by sample) into NCHW and adds the per-channel bias.
    extern "C" __global__ void conv2d_col2out(
        float* out, const float* mm_out, const float* bias,
        int N, int c_out, int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int col_w = out_h * out_w;
        int total = N * c_out * col_w;
        if (idx >= total) return;

        int p  = idx % col_w;
        int oc = (idx / col_w) % c_out;
        int ni = idx / (col_w * c_out);

        out[idx] = mm_out[oc * (N * col_w) + ni * col_w + p] + bias[oc];
    }

    extern "C" __global__ void conv2d_bias_grad(
        float* dbias, const float* grad, int N, int c_out, int out_h, int out_w
    ) {
        int oc = blockIdx.x * blockDim.x + threadIdx.x;
        if (oc >= c_out) return;

        float acc = 0.0f;
        for (int ni = 0; ni < N; ++ni)
            for (int oy = 0; oy < out_h; ++oy)
                for (int ox = 0; ox < out_w; ++ox)
                    acc += grad[((ni * c_out + oc) * out_h + oy) * out_w + ox];
        dbias[oc] = acc;
    }

    extern "C" __global__ void conv2d_weight_grad(
        float* dweight, const float* grad, const float* input,
        int N, int c_in, int in_h, int in_w,
        int c_out, int kh, int kw, int stride, int pad,
        int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int wsize = c_out * c_in * kh * kw;
        int total = wsize * N;
        if (idx >= total) return;

        int w  = idx % wsize;
        int ni = idx / wsize;

        int j  = w % kw;
        int i  = (w / kw) % kh;
        int ic = (w / (kw * kh)) % c_in;
        int oc = w / (kw * kh * c_in);

        float acc = 0.0f;
        for (int oy = 0; oy < out_h; ++oy)
            for (int ox = 0; ox < out_w; ++ox) {
                int iy = oy * stride + i - pad;
                int ix = ox * stride + j - pad;
                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                    int g_idx  = ((ni * c_out + oc) * out_h + oy) * out_w + ox;
                    int in_idx = ((ni * c_in + ic) * in_h + iy) * in_w + ix;
                    acc += grad[g_idx] * input[in_idx];
                }
            }
        atomicAdd(&dweight[w], acc);
    }

    extern "C" __global__ void conv2d_input_grad(
        float* dinput, const float* grad, const float* weight,
        int N, int c_in, int in_h, int in_w,
        int c_out, int kh, int kw, int stride, int pad,
        int out_h, int out_w
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * c_in * in_h * in_w;
        if (idx >= total) return;

        int x  = idx % in_w;
        int y  = (idx / in_w) % in_h;
        int ic = (idx / (in_w * in_h)) % c_in;
        int ni = idx / (in_w * in_h * c_in);

        float acc = 0.0f;
        for (int oc = 0; oc < c_out; ++oc)
            for (int i = 0; i < kh; ++i) {
                int oy_num = y + pad - i;
                if (oy_num < 0 || oy_num % stride != 0) continue;
                int oy = oy_num / stride;
                if (oy >= out_h) continue;

                for (int j = 0; j < kw; ++j) {
                    int ox_num = x + pad - j;
                    if (ox_num < 0 || ox_num % stride != 0) continue;
                    int ox = ox_num / stride;
                    if (ox >= out_w) continue;

                    int g_idx = ((ni * c_out + oc) * out_h + oy) * out_w + ox;
                    int w_idx = ((oc * c_in + ic) * kh + i) * kw + j;
                    acc += grad[g_idx] * weight[w_idx];
                }
            }
        dinput[idx] = acc;
    }
"#;

crate::kernel_module!(KERNEL);

#[derive(Clone, Copy)]
struct Dims {
    n: usize, c_in: usize, in_h: usize, in_w: usize,
    c_out: usize, kh: usize, kw: usize, stride: usize, pad: usize,
    out_h: usize, out_w: usize,
}

fn launch_im2col(input: &CudaSlice<f32>, d: Dims) -> CudaSlice<f32> {
    let stream = backend::stream();
    let col_h = d.c_in * d.kh * d.kw;
    let width = d.n * d.out_h * d.out_w;
    let mut cols = stream.alloc_zeros::<f32>(col_h * width).expect("conv2d im2col: alloc failed");
    let f = module().load_function("conv2d_im2col").expect("conv2d_im2col not found");
    let cfg = LaunchConfig::for_num_elems((col_h * width) as u32);
    let (n, cin, ih, iw) = (d.n as i32, d.c_in as i32, d.in_h as i32, d.in_w as i32);
    let (kh, kw, st, pd) = (d.kh as i32, d.kw as i32, d.stride as i32, d.pad as i32);
    let (oh, ow) = (d.out_h as i32, d.out_w as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&mut cols); b.arg(input);
    b.arg(&n); b.arg(&cin); b.arg(&ih); b.arg(&iw);
    b.arg(&kh); b.arg(&kw); b.arg(&st); b.arg(&pd);
    b.arg(&oh); b.arg(&ow);
    unsafe { b.launch(cfg).expect("conv2d im2col: launch failed"); }
    cols
}

fn launch_col2out(mm_out: &CudaSlice<f32>, bias: &CudaSlice<f32>, d: Dims) -> CudaSlice<f32> {
    let stream = backend::stream();
    let total = d.n * d.c_out * d.out_h * d.out_w;
    let mut out = stream.alloc_zeros::<f32>(total).expect("conv2d col2out: alloc failed");
    let f = module().load_function("conv2d_col2out").expect("conv2d_col2out not found");
    let cfg = LaunchConfig::for_num_elems(total as u32);
    let (n, co, oh, ow) = (d.n as i32, d.c_out as i32, d.out_h as i32, d.out_w as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&mut out); b.arg(mm_out); b.arg(bias);
    b.arg(&n); b.arg(&co); b.arg(&oh); b.arg(&ow);
    unsafe { b.launch(cfg).expect("conv2d col2out: launch failed"); }
    out
}

fn launch_bias_grad(grad: &CudaSlice<f32>, d: Dims) -> CudaSlice<f32> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(d.c_out).expect("conv2d bgrad: alloc failed");
    let f = module().load_function("conv2d_bias_grad").expect("conv2d_bias_grad not found");
    let cfg = LaunchConfig::for_num_elems(d.c_out as u32);
    let (n, co, oh, ow) = (d.n as i32, d.c_out as i32, d.out_h as i32, d.out_w as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&mut out); b.arg(grad); b.arg(&n); b.arg(&co); b.arg(&oh); b.arg(&ow);
    unsafe { b.launch(cfg).expect("conv2d bgrad: launch failed"); }
    out
}

fn launch_weight_grad(grad: &CudaSlice<f32>, input: &CudaSlice<f32>, d: Dims) -> CudaSlice<f32> {
    let stream = backend::stream();
    let wsize = d.c_out * d.c_in * d.kh * d.kw;
    let mut out = stream.alloc_zeros::<f32>(wsize).expect("conv2d wgrad: alloc failed");
    let f = module().load_function("conv2d_weight_grad").expect("conv2d_weight_grad not found");
    let cfg = LaunchConfig::for_num_elems((wsize * d.n) as u32);  // one thread per (weight, sample)
    let (n, cin, ih, iw) = (d.n as i32, d.c_in as i32, d.in_h as i32, d.in_w as i32);
    let (co, kh, kw, st, pd) = (d.c_out as i32, d.kh as i32, d.kw as i32, d.stride as i32, d.pad as i32);
    let (oh, ow) = (d.out_h as i32, d.out_w as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&mut out); b.arg(grad); b.arg(input);
    b.arg(&n); b.arg(&cin); b.arg(&ih); b.arg(&iw);
    b.arg(&co); b.arg(&kh); b.arg(&kw); b.arg(&st); b.arg(&pd);
    b.arg(&oh); b.arg(&ow);
    unsafe { b.launch(cfg).expect("conv2d wgrad: launch failed"); }
    out
}

fn launch_input_grad(grad: &CudaSlice<f32>, weight: &CudaSlice<f32>, d: Dims) -> CudaSlice<f32> {
    let stream = backend::stream();
    let total = d.n * d.c_in * d.in_h * d.in_w;
    let mut out = stream.alloc_zeros::<f32>(total).expect("conv2d igrad: alloc failed");
    let f = module().load_function("conv2d_input_grad").expect("conv2d_input_grad not found");
    let cfg = LaunchConfig::for_num_elems(total as u32);
    let (n, cin, ih, iw) = (d.n as i32, d.c_in as i32, d.in_h as i32, d.in_w as i32);
    let (co, kh, kw, st, pd) = (d.c_out as i32, d.kh as i32, d.kw as i32, d.stride as i32, d.pad as i32);
    let (oh, ow) = (d.out_h as i32, d.out_w as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&mut out); b.arg(grad); b.arg(weight);
    b.arg(&n); b.arg(&cin); b.arg(&ih); b.arg(&iw);
    b.arg(&co); b.arg(&kh); b.arg(&kw); b.arg(&st); b.arg(&pd);
    b.arg(&oh); b.arg(&ow);
    unsafe { b.launch(cfg).expect("conv2d igrad: launch failed"); }
    out
}

/// 2-D convolution of a resident input with resident weight/bias, kept on the
/// GPU. Input `[N, c_in, in_h, in_w]`, weight `[c_out, c_in, kh, kw]`, bias
/// `[c_out]`; output `[N, c_out, out_h, out_w]`. Zero-padding `pad`, `stride`.
///
/// # Panics
/// If any input is not on the GPU.
pub fn conv2d(
    input: &Rc<RefCell<Node>>,
    weight: &Rc<RefCell<Node>>,
    bias: &Rc<RefCell<Node>>,
    stride: usize,
    pad: usize,
) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, d) = {
        let in_n = input.borrow();
        let w_n = weight.borrow();
        let b_n = bias.borrow();
        let in_gpu = in_n.gpu.as_ref().expect("cuda conv2d: input not on GPU");
        let w_gpu = w_n.gpu.as_ref().expect("cuda conv2d: weight not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda conv2d: bias not on GPU");

        let (n, c_in, in_h, in_w) = (in_n.shape[0], in_n.shape[1], in_n.shape[2], in_n.shape[3]);
        let (c_out, kh, kw) = (w_n.shape[0], w_n.shape[2], w_n.shape[3]);
        let out_h = (in_h + 2 * pad - kh) / stride + 1;
        let out_w = (in_w + 2 * pad - kw) / stride + 1;
        let d = Dims { n, c_in, in_h, in_w, c_out, kh, kw, stride, pad, out_h, out_w };
        
        let cols = launch_im2col(&in_gpu.data, d);
        let mm_out = mm(
            "matmul_nn",
            &w_gpu.data,
            &cols,
            c_out,
            n * out_h * out_w,
            c_in * kh * kw,
        );
        let out = launch_col2out(&mm_out, &b_gpu.data, d);

        (out, d)
    };

    let out_node = Node::new(vec![], vec![d.n, d.c_out, d.out_h, d.out_w]);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone(), weight.clone(), bias.clone()];

        let len: usize = node.shape.iter().product();
        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda conv2d: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        let w_bwd = weight.clone();
        let b_bwd = bias.clone();
        let in_len = d.n * d.c_in * d.in_h * d.in_w;
        let w_len = d.c_out * d.c_in * d.kh * d.kw;

        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let og = grad.borrow();

            let di = {
                let w = w_bwd.borrow();
                let w_data = &w.gpu.as_ref().expect("conv2d bwd: weight not on GPU").data;
                launch_input_grad(&og, w_data, d)
            };
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(di)), in_len);

            let dw = {
                let inp = in_bwd.borrow();
                let in_data = &inp.gpu.as_ref().expect("conv2d bwd: input not on GPU").data;
                launch_weight_grad(&og, in_data, d)
            };
            accumulate_into(&w_bwd, &Rc::new(RefCell::new(dw)), w_len);

            let db = launch_bias_grad(&og, d);
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(db)), d.c_out);
        }));
    }

    out_node
}