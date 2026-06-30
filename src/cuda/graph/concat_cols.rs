//! Column concatenation of several resident nodes (autograd op). Backward splits
//! grad back to each part. Inverse of slice_cols.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void concat_write(float* out, const float* part, int rows, int total_cols, int col_offset, int pw) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = rows * pw;
        if(t < total) {
            int r = t / pw;
            int c = t % pw;
            out[r * total_cols + col_offset + c] = part[r * pw + c];
        }
    }
    extern "C" __global__ void concat_read(float* temp, const float* grad, int rows, int total_cols, int col_offset, int pw) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = rows * pw;
        if(t < total) {
            int r = t / pw;
            int c = t % pw;
            temp[r * pw + c] = grad[r * total_cols + col_offset + c];
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Concatenates resident nodes side by side along the last dim. All parts must
/// share the same number of rows.
///
/// # Panics
/// If any part is not on the GPU.
pub fn concat_cols(parts: &[Rc<RefCell<Node>>]) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let widths: Vec<usize> = parts.iter().map(|p| *p.borrow().shape.last().unwrap()).collect();
    let total_cols: usize = widths.iter().sum();
    let rows = {
        let p0 = parts[0].borrow();
        let len: usize = p0.shape.iter().product();
        len / widths[0]
    };
    let out_len = rows * total_cols;

    let mut out = stream.alloc_zeros::<f32>(out_len).expect("cuda concat_cols: alloc failed");
    let fw = module().load_function("concat_write").expect("concat_write not found");
    let mut col_offset = 0usize;
    for (idx, part) in parts.iter().enumerate() {
        let p = part.borrow();
        let pd = &p.gpu.as_ref().expect("cuda concat_cols: part not on GPU").data;
        let pw = widths[idx];
        let (rr, tc, co, pww) = (rows as i32, total_cols as i32, col_offset as i32, pw as i32);
        let cfg = LaunchConfig::for_num_elems((rows * pw) as u32);
        let mut builder = stream.launch_builder(&fw);
        builder.arg(&mut out); builder.arg(pd); builder.arg(&rr); builder.arg(&tc); builder.arg(&co); builder.arg(&pww);
        unsafe { builder.launch(cfg).expect("cuda concat_cols: write launch failed"); }
        col_offset += pw;
    }

    let mut out_shape = parts[0].borrow().shape.clone();
    let last = out_shape.len() - 1;
    out_shape[last] = total_cols;

    let out_node = Node::new(vec![], out_shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = parts.to_vec();
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(out_len).expect("cuda concat_cols: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out, grad: grad.clone() });
        let parts_bwd = parts.to_vec();
        let widths_bwd = widths.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let fr = module().load_function("concat_read").expect("concat_read not found");
            let mut col_offset = 0usize;
            for (idx, part) in parts_bwd.iter().enumerate() {
                let pw = widths_bwd[idx];
                let part_len = rows * pw;
                let mut temp = stream.alloc_zeros::<f32>(part_len).expect("cuda concat_cols bwd: alloc");
                let (rr, tc, co, pww) = (rows as i32, total_cols as i32, col_offset as i32, pw as i32);
                let cfg = LaunchConfig::for_num_elems(part_len as u32);
                let mut builder = stream.launch_builder(&fr);
                builder.arg(&mut temp); builder.arg(&*g); builder.arg(&rr); builder.arg(&tc); builder.arg(&co); builder.arg(&pww);
                unsafe { builder.launch(cfg).expect("cuda concat_cols bwd: read launch failed"); }
                accumulate_into(part, &Rc::new(RefCell::new(temp)), part_len);
                col_offset += pw;
            }
        }));
    }
    out_node
}