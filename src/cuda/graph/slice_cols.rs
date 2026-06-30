//! Column slice [col_start, col_end) of a resident node's last dim (autograd op).
//! Backward scatters grad back into the original columns (zeros elsewhere).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void slice_fwd(float* out, const float* in, int rows, int total_cols, int col_start, int slice_w) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = rows * slice_w;
        if(t < total) {
            int r = t / slice_w;
            int c = t % slice_w;
            out[r * slice_w + c] = in[r * total_cols + col_start + c];
        }
    }
    extern "C" __global__ void slice_bwd(float* temp, const float* grad, int rows, int total_cols, int col_start, int slice_w) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = rows * slice_w;
        if(t < total) {
            int r = t / slice_w;
            int c = t % slice_w;
            temp[r * total_cols + col_start + c] = grad[r * slice_w + c];
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Takes columns `[col_start, col_end)` from a resident node's last dim, all rows.
///
/// # Panics
/// If the input is not on the GPU.
pub fn slice_cols(a: &Rc<RefCell<Node>>, col_start: usize, col_end: usize) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let slice_w = col_end - col_start;
    let (out_data, out_shape, rows, total_cols, out_len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda slice_cols: input not on GPU");
        let shape = &a_n.shape;
        let total_cols = *shape.last().unwrap();
        let in_len: usize = shape.iter().product();
        let rows = in_len / total_cols;
        let out_len = rows * slice_w;

        let mut out = stream.alloc_zeros::<f32>(out_len).expect("cuda slice_cols: alloc failed");
        let f = module().load_function("slice_fwd").expect("slice_fwd not found");
        let (rr, tc, cs, sw) = (rows as i32, total_cols as i32, col_start as i32, slice_w as i32);
        let cfg = LaunchConfig::for_num_elems(out_len as u32);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut out); builder.arg(&a_gpu.data); builder.arg(&rr); builder.arg(&tc); builder.arg(&cs); builder.arg(&sw);
        unsafe { builder.launch(cfg).expect("cuda slice_cols: launch failed"); }

        let mut out_shape = shape.clone();
        let last = out_shape.len() - 1;
        out_shape[last] = slice_w;
        (out, out_shape, rows, total_cols, out_len)
    };
    let out_node = Node::new(vec![], out_shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(out_len).expect("cuda slice_cols: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        let in_len = rows * total_cols;
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let mut temp = stream.alloc_zeros::<f32>(in_len).expect("cuda slice_cols bwd: alloc"); // zeros outside slice
            let f = module().load_function("slice_bwd").expect("slice_bwd not found");
            let (rr, tc, cs, sw) = (rows as i32, total_cols as i32, col_start as i32, slice_w as i32);
            let cfg = LaunchConfig::for_num_elems(out_len as u32);
            let mut builder = stream.launch_builder(&f);
            builder.arg(&mut temp); builder.arg(&*g); builder.arg(&rr); builder.arg(&tc); builder.arg(&cs); builder.arg(&sw);
            unsafe { builder.launch(cfg).expect("cuda slice_cols bwd: launch"); }
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(temp)), in_len);
        }));
    }
    out_node
}