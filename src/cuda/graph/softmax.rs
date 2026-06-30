//! Softmax over the last dim (resident autograd op). Backward uses the Jacobian
//! trick grad_in = out * (grad - sum(out*grad)), recomputing softmax from input.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void softmax_forward(float* out, const float* in, int rows, int width) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if(r < rows) {
            int start = r * width;
            float max = in[start];
            for(int f = 1; f < width; ++f) max = fmaxf(max, in[start + f]);
            float sum = 0.0f;
            for(int f = 0; f < width; ++f) sum += expf(in[start + f] - max);
            for(int f = 0; f < width; ++f) out[start + f] = expf(in[start + f] - max) / sum;
        }
    }
    extern "C" __global__ void softmax_backward(float* gin, const float* in, const float* grad, int rows, int width) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if(r < rows) {
            int start = r * width;
            float max = in[start];
            for(int f = 1; f < width; ++f) max = fmaxf(max, in[start + f]);
            float sum = 0.0f;
            for(int f = 0; f < width; ++f) sum += expf(in[start + f] - max);
            float dot = 0.0f;
            for(int f = 0; f < width; ++f) { float sm = expf(in[start + f] - max) / sum; dot += sm * grad[start + f]; }
            for(int f = 0; f < width; ++f) { float sm = expf(in[start + f] - max) / sum; gin[start + f] = sm * (grad[start + f] - dot); }
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Softmax over the last dim of a resident node, kept on the GPU.
///
/// # Panics
/// If the input is not on the GPU.
pub fn softmax(a: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, rows, width, len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda softmax: input not on GPU");
        let shape = a_n.shape.clone();
        let width = *shape.last().unwrap();
        let len: usize = shape.iter().product();
        let rows = len / width;

        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda softmax: alloc failed");
        let f = module().load_function("softmax_forward").expect("softmax_forward not found");
        let (rr, ww) = (rows as i32, width as i32);
        let cfg = LaunchConfig::for_num_elems(rows as u32);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut out); builder.arg(&a_gpu.data); builder.arg(&rr); builder.arg(&ww);
        unsafe { builder.launch(cfg).expect("cuda softmax: launch failed"); }
        (out, shape, rows, width, len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda softmax: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let local = {
                let a_n = a_bwd.borrow();
                let in_d = &a_n.gpu.as_ref().expect("cuda softmax bwd: input not on GPU").data;
                let mut temp = stream.alloc_zeros::<f32>(len).expect("cuda softmax bwd: alloc");
                let f = module().load_function("softmax_backward").expect("softmax_backward not found");
                let (rr, ww) = (rows as i32, width as i32);
                let cfg = LaunchConfig::for_num_elems(rows as u32);
                let mut builder = stream.launch_builder(&f);
                builder.arg(&mut temp); builder.arg(in_d); builder.arg(&*g); builder.arg(&rr); builder.arg(&ww);
                unsafe { builder.launch(cfg).expect("cuda softmax bwd: launch"); }
                temp
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(local)), len);
        }));
    }
    out_node
}