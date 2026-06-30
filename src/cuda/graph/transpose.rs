//! Transpose of the last two dims (resident autograd op), batched over leading
//! dims. [..., rows, cols] -> [..., cols, rows]. Backward transposes grad back.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void transpose2d(float* out, const float* in, int batch, int R, int C) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * R * C;
        if(t < total) {
            int b = t / (R * C);
            int rem = t % (R * C);
            int i = rem / C;
            int j = rem % C;
            int off = b * R * C;
            out[off + j * R + i] = in[off + i * C + j];
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Transposes the last two dims of a resident node, batched over leading dims.
///
/// # Panics
/// If the input is not on the GPU.
pub fn transpose(a: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, out_shape, batch, rows, cols, len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda transpose: input not on GPU");
        let shape = &a_n.shape;
        let ndim = shape.len();
        let rows = shape[ndim - 2];
        let cols = shape[ndim - 1];
        let len: usize = shape.iter().product();
        let batch = len / (rows * cols);

        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda transpose: alloc failed");
        let f = module().load_function("transpose2d").expect("transpose2d not found");
        let (bb, rr, cc) = (batch as i32, rows as i32, cols as i32);
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut out); builder.arg(&a_gpu.data); builder.arg(&bb); builder.arg(&rr); builder.arg(&cc);
        unsafe { builder.launch(cfg).expect("cuda transpose: launch failed"); }

        let mut out_shape = shape.clone();
        out_shape[ndim - 2] = cols;
        out_shape[ndim - 1] = rows;
        (out, out_shape, batch, rows, cols, len)
    };
    let out_node = Node::new(vec![], out_shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda transpose: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let mut temp = stream.alloc_zeros::<f32>(len).expect("cuda transpose bwd: alloc");
            let f = module().load_function("transpose2d").expect("transpose2d not found");
            let (bb, rr, cc) = (batch as i32, cols as i32, rows as i32); // swap: grad is [.., cols, rows]
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut builder = stream.launch_builder(&f);
            builder.arg(&mut temp); builder.arg(&*g); builder.arg(&bb); builder.arg(&rr); builder.arg(&cc);
            unsafe { builder.launch(cfg).expect("cuda transpose bwd: launch"); }
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(temp)), len);
        }));
    }
    out_node
}