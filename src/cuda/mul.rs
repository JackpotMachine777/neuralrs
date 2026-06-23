//! Element-wise multiply on the GPU (resident autograd op): out = a * b.
//!
//! Forward runs `vmul` on data already in VRAM. Backward follows the product
//! rule: the gradient to each parent is the upstream gradient times the *other*
//! operand, accumulated via the shared accumulate primitive.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::backend;
use super::graph::accumulate_into;
use crate::autograd::node::{GpuBuffers, Node};

const KERNEL: &str = r#"
    extern "C" __global__ void vmul(float* out, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) out[i] = a[i] * b[i];
    }
"#;

crate::kernel_module!(KERNEL);

/// Computes `grad * other.data` into a fresh device buffer, reusing the `vmul`
/// kernel. This is one parent's gradient before it's accumulated in.
fn grad_times(
    grad: &Rc<RefCell<CudaSlice<f32>>>,
    other: &Rc<RefCell<Node>>,
    n: usize,
) -> Rc<RefCell<CudaSlice<f32>>> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(n).expect("cuda mul: backward alloc failed");

    let g = grad.borrow();
    let o = other.borrow();
    let other_data = &o.gpu.as_ref().expect("cuda mul: operand not on GPU").data;

    let vmul = module().load_function("vmul").expect("vmul not found");
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut builder = stream.launch_builder(&vmul);
    builder.arg(&mut out);
    builder.arg(&*g);
    builder.arg(other_data);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("cuda mul: backward launch failed"); }

    Rc::new(RefCell::new(out))
}

/// Element-wise product of two resident nodes: `out = a * b`, computed and kept
/// on the GPU. The result is also resident (its CPU `data` stays empty).
///
/// # Panics
/// If either input is not on the GPU.
pub fn mul(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda mul: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda mul: rhs not on GPU");

        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda mul: device alloc failed");

        let vmul = module().load_function("vmul").expect("vmul not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);

        let mut builder = stream.launch_builder(&vmul);
        builder.arg(&mut out);
        builder.arg(&a_gpu.data);
        builder.arg(&b_gpu.data);
        builder.arg(&len);
        unsafe { builder.launch(cfg).expect("cuda mul: launch failed"); }

        (out, a_n.shape.clone(), len)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone(), b.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda mul: grad alloc failed"),
        ));
        n.gpu = Some(GpuBuffers {
            data: out_data,
            grad: grad.clone(),
        });

        let a_bwd = a.clone();
        let b_bwd = b.clone();

        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let ga = grad_times(&grad, &b_bwd, len);
            accumulate_into(&a_bwd, &ga, len);

            let gb = grad_times(&grad, &a_bwd, len);
            accumulate_into(&b_bwd, &gb, len);
        }));
    }

    out_node
}