//! ReLU on the GPU (resident autograd op): out = max(0, x).
//!
//! Forward runs `vrelu`. Backward passes the gradient through wherever the input
//! was positive and blocks it elsewhere (d/dx max(0,x) = 1 if x > 0 else 0),
//! accumulating into the input via the shared accumulate primitive.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vrelu(float* out, const float* x, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }

    extern "C" __global__ void relu_grad(float* out, const float* grad, const float* x, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) out[i] = x[i] > 0.0f? grad[i] : 0.0f;
    }
"#;

crate::kernel_module!(KERNEL);

/// Computes the masked upstream gradient (`grad` where `x > 0`, else 0) into a
/// fresh device buffer, the input's gradient before it's accumulated in.
fn masked_grad(
    grad: &Rc<RefCell<CudaSlice<f32>>>,
    x: &Rc<RefCell<Node>>,
    n: usize,
) -> Rc<RefCell<CudaSlice<f32>>> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(n).expect("cuda relu: backward alloc failed");

    let g = grad.borrow();
    let xn = x.borrow();
    let x_data = &xn.gpu.as_ref().expect("cuda relu: input not on GPU").data;

    let f = module().load_function("relu_grad").expect("relu_grad not found");
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut out);
    builder.arg(&*g);
    builder.arg(x_data);
    builder.arg(&n);
    unsafe { builder.launch(cfg).expect("cuda relu: backward launch failed"); }

    Rc::new(RefCell::new(out))
}

/// ReLU of a resident node: `out = max(0, x)`, computed and kept on the GPU.
/// The result is also resident (its CPU `data` stays empty).
///
/// # Panics
/// If the input is not on the GPU.
pub fn relu(x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, shape, len) = {
        let xn = x.borrow();
        let x_gpu = xn.gpu.as_ref().expect("cuda relu: input not on GPU");

        let len: usize = xn.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda relu: device alloc failed");

        let vrelu = module().load_function("vrelu").expect("vrelu not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut builder = stream.launch_builder(&vrelu);
        builder.arg(&mut out);
        builder.arg(&x_gpu.data);
        builder.arg(&len);
        unsafe { builder.launch(cfg).expect("cuda relu: launch failed"); }

        (out, xn.shape.clone(), len)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![x.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda relu: grad alloc failed"),
        ));
        n.gpu = Some(GpuBuffers {
            data: out_data,
            grad: grad.clone(),
        });

        let x_bwd = x.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let g = masked_grad(&grad, &x_bwd, len);
            accumulate_into(&x_bwd, &g, len);
        }));
    }

    out_node
}