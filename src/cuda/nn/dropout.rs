//! Dropout on the GPU (resident autograd op).
//!
//! In training, zeroes each element with probability `p` and scales the
//! survivors by 1/(1-p) (inverted dropout), mirroring the CPU `Dropout`. The
//! random mask (0 or scale per element) is drawn on the host and uploaded; the
//! same mask routes the gradient in backward. In eval it's a passthrough.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void mask_mul(float* out, const float* a, const float* mask, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = a[i] * mask[i];
    }
"#;

crate::kernel_module!(KERNEL);

/// Dropout of a resident input. In training each element is zeroed with
/// probability `p` and survivors are scaled by 1/(1-p); the result is resident.
/// In eval (`training == false`) the input is passed through unchanged.
///
/// # Panics
/// If the input is not on the GPU (training mode).
pub fn dropout(input: &Rc<RefCell<Node>>, p: f32, training: bool) -> Rc<RefCell<Node>> {
    if !training {
        return input.clone();
    }

    let stream = backend::stream();
    let scale = 1.0 / (1.0 - p);

    let (out_data, mask, shape, len) = {
        let in_n = input.borrow();
        let in_gpu = in_n.gpu.as_ref().expect("cuda dropout: input not on GPU");
        let len: usize = in_n.shape.iter().product();

        let mask_host: Vec<f32> = (0..len)
            .map(|_| if rand::random::<f32>() < p { 0.0 } else { scale })
            .collect();
        let mask = stream.clone_htod(&mask_host).expect("cuda dropout: mask htod failed");

        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda dropout: out alloc failed");
        let f = module().load_function("mask_mul").expect("mask_mul not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut b = stream.launch_builder(&f);
        b.arg(&mut out); b.arg(&in_gpu.data); b.arg(&mask); b.arg(&len);
        unsafe { b.launch(cfg).expect("cuda dropout: launch failed"); }

        (out, mask, in_n.shape.clone(), len)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(len).expect("cuda dropout: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();
            let mut temp = stream.alloc_zeros::<f32>(len).expect("cuda dropout bwd: alloc failed");
            let f = module().load_function("mask_mul").expect("mask_mul not found");
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut temp); b.arg(&*og); b.arg(&mask); b.arg(&len);
            unsafe { b.launch(cfg).expect("cuda dropout bwd: launch failed"); }
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(temp)), len);
        }));
    }

    out_node
}