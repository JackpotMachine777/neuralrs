//! Element-wise add on the GPU (resident autograd op): out = a + b.
//!
//! Two modes, mirroring the CPU `add`: same-shape adds straight across, and a
//! 1-D `[features]` bias added to a 2-D `[batch, features]` input broadcasts over
//! the batch. Backward passes the gradient 1:1 to a same-shape parent; for the
//! bias it sums the gradient over the batch (it was reused for every row).

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::reduce;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vadd(float* out, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n) out[i] = a[i] + b[i];
    }

    extern "C" __global__ void bias_add(float* out, const float* a, const float* bias, int rows, int cols) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int n = rows * cols;

        if(i < n) out[i] = a[i] + bias[i % cols];
    }
"#;

crate::kernel_module!(KERNEL);

/// Element-wise add of two resident nodes: `out = a + b`, computed and kept on
/// the GPU (result resident, its CPU `data` stays empty). A 1-D bias added to a
/// 2-D `[batch, features]` input broadcasts over the batch.
///
/// # Panics
/// If either input is not on the GPU.
pub fn add(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, shape, len, broadcast) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda add: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda add: rhs not on GPU");

        let broadcast = a_n.shape.len() == 2 && b_n.shape.len() == 1 && a_n.shape[1] == b_n.shape[0];
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda add: device alloc failed");
        let cfg = LaunchConfig::for_num_elems(len as u32);

        if broadcast {
            let (rows, cols) = (a_n.shape[0] as i32, a_n.shape[1] as i32);
            let f = module().load_function("bias_add").expect("bias_add not found");
            let mut builder = stream.launch_builder(&f);
            builder.arg(&mut out);
            builder.arg(&a_gpu.data);
            builder.arg(&b_gpu.data);
            builder.arg(&rows);
            builder.arg(&cols);
            unsafe { builder.launch(cfg).expect("cuda add: bias_add launch failed"); }
        } else {
            let f = module().load_function("vadd").expect("vadd not found");
            let mut builder = stream.launch_builder(&f);
            builder.arg(&mut out);
            builder.arg(&a_gpu.data);
            builder.arg(&b_gpu.data);
            builder.arg(&len);
            unsafe { builder.launch(cfg).expect("cuda add: launch failed"); }
        }

        (out, a_n.shape.clone(), len, broadcast)
    };

    let out_node = Node::new(vec![], shape.clone());
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone(), b.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda add: grad alloc failed"),
        ));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let a_bwd = a.clone();
        let b_bwd = b.clone();

        if broadcast {
            let (rows, cols) = (shape[0], shape[1]);
            n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
                accumulate_into(&a_bwd, &grad, len);
                let db = {
                    let g = grad.borrow();
                    reduce::sum_axis0(&g, rows, cols)
                };
                accumulate_into(&b_bwd, &Rc::new(RefCell::new(db)), cols);
            }));
        } else {
            n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
                accumulate_into(&a_bwd, &grad, len);
                accumulate_into(&b_bwd, &grad, len);
            }));
        }
    }

    out_node
}