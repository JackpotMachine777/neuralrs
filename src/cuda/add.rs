//! Element-wise add on the GPU (resident autograd op): out = a + b.
//!
//! Forward runs `vadd` on data already in VRAM. Backward passes the gradient
//! straight through to both parents via the shared accumulate primitive.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaModule, LaunchConfig, PushKernelArg};

use super::backend;
use super::graph::accumulate_into;
use crate::autograd::node::{GpuBuffers, Node};

const KERNEL: &str = r#"
    extern "C" __global__ void vadd(float* out, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) out[i] = a[i] + b[i];
    }
"#;

static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();

fn module() -> &'static Arc<CudaModule> {
    MODULE.get_or_init(|| backend::compile(KERNEL))
}

/// Element-wise add of two resident nodes: `out = a + b`, computed and kept on
/// the GPU. The result is also resident (its CPU `data` stays empty).
///
/// Forward only for now, the backward pass is added next.
///
/// # Panics
/// If either input is not on the GPU.
pub fn add(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda add: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda add: rhs not on GPU");

        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda add: device alloc failed");

        let vadd = module().load_function("vadd").expect("vadd not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);

        let mut builder = stream.launch_builder(&vadd);
        builder.arg(&mut out);
        builder.arg(&a_gpu.data);
        builder.arg(&b_gpu.data);
        builder.arg(&len);
        unsafe { builder.launch(cfg).expect("cuda add: launch failed"); }

        (out, a_n.shape.clone(), len)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone(), b.clone()];
        
        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda add: grad alloc failed"),
        ));
        n.gpu = Some(GpuBuffers {
            data: out_data,
            grad: grad.clone(),
        });

        let a_bwd = a.clone();
        let b_bwd = b.clone();

        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            accumulate_into(&a_bwd, &grad, len);
            accumulate_into(&b_bwd, &grad, len);
        }));
    }

    out_node
}