//! Reshape on the GPU (resident autograd op): reinterpret under a new shape.
//!
//! The element count must match (asserted). Reshape only relabels the layout,
//! the contiguous data is unchanged, so the forward copies the buffer as-is and
//! backward passes the gradient straight through 1:1 (same linear order).

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void copy(float* out, const float* in, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) out[i] = in[i];
    }
"#;

crate::kernel_module!(KERNEL);

/// Reinterprets a resident node under `new_shape`, keeping the same values. The
/// result is resident; data is copied on the device (no host round-trip).
///
/// # Panics
/// If the input is not on the GPU, or the element count doesn't match.
pub fn reshape(x: &Rc<RefCell<Node>>, new_shape: Vec<usize>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, len) = {
        let x_n = x.borrow();
        let x_gpu = x_n.gpu.as_ref().expect("cuda reshape: input not on GPU");

        let len: usize = x_n.shape.iter().product();
        let new_len: usize = new_shape.iter().product();
        assert_eq!(len, new_len, "cuda reshape: element count mismatched");

        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda reshape: device alloc failed");

        let f = module().load_function("copy").expect("copy not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut out);
        builder.arg(&x_gpu.data);
        builder.arg(&len);
        unsafe { builder.launch(cfg).expect("cuda reshape: launch failed"); }

        (out, len)
    };

    let out_node = Node::new(vec![], new_shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![x.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda reshape: grad alloc failed"),
        ));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let x_bwd = x.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            accumulate_into(&x_bwd, &grad, len);
        }));
    }

    out_node
}