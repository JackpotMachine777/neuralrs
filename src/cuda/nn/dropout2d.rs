//! Spatial dropout on the GPU (resident autograd op).
//!
//! Drops entire channels at once during training (instead of individual
//! pixels), mirroring the CPU `Dropout2d`: a per-(sample, channel) mask
//! (0 or 1/(1-p)) is drawn on the host and uploaded, then broadcast over
//! both spatial dimensions; the same mask routes the gradient in backward.
//! In eval it's a passthrough.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    // The mask has one entry per (sample, channel); element i belongs to mask
    // slot i / hw, so every pixel of a channel shares one factor.
    extern "C" __global__ void mask_mul_channels(
        float* out, const float* a, const float* mask, int total, int hw
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total) out[i] = a[i] * mask[i / hw];
    }
"#;

crate::kernel_module!(KERNEL);

/// Spatial dropout of a resident 4-D input `[batch, channels, h, w]`. In
/// training each (sample, channel) plane is zeroed whole with probability `p`
/// and surviving channels are scaled by 1/(1-p); the result is resident. In
/// eval (`training == false`) the input is passed through unchanged.
///
/// # Panics
/// If the input is not on the GPU (training mode).
pub fn dropout2d(input: &Rc<RefCell<Node>>, p: f32, training: bool) -> Rc<RefCell<Node>> {
    if !training {
        return input.clone();
    }

    let stream = backend::stream();
    let scale = 1.0 / (1.0 - p);

    let (out_data, mask, shape, total, hw) = {
        let in_n = input.borrow();
        let in_gpu = in_n.gpu.as_ref().expect("cuda dropout2d: input not on GPU");
        let (n, c) = (in_n.shape[0], in_n.shape[1]);
        let hw = in_n.shape[2] * in_n.shape[3];
        let total = n * c * hw;

        let mask_host: Vec<f32> = (0..n * c)
            .map(|_| if rand::random::<f32>() < p { 0.0 } else { scale })
            .collect();
        let mask = stream.clone_htod(&mask_host).expect("cuda dropout2d: mask htod failed");

        let mut out = stream.alloc_zeros::<f32>(total).expect("cuda dropout2d: out alloc failed");
        let f = module().load_function("mask_mul_channels").expect("mask_mul_channels not found");
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let (ti, hwi) = (total as i32, hw as i32);
        let mut b = stream.launch_builder(&f);
        b.arg(&mut out); b.arg(&in_gpu.data); b.arg(&mask); b.arg(&ti); b.arg(&hwi);
        unsafe { b.launch(cfg).expect("cuda dropout2d: launch failed"); }

        (out, mask, in_n.shape.clone(), total, hw)
    };

    let out_node = Node::new(vec![], shape);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![input.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(total).expect("cuda dropout2d: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let in_bwd = input.clone();
        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let og = grad.borrow();
            let mut temp = stream.alloc_zeros::<f32>(total).expect("cuda dropout2d bwd: alloc failed");
            let f = module().load_function("mask_mul_channels").expect("mask_mul_channels not found");
            let cfg = LaunchConfig::for_num_elems(total as u32);
            let (ti, hwi) = (total as i32, hw as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut temp); b.arg(&*og); b.arg(&mask); b.arg(&ti); b.arg(&hwi);
            unsafe { b.launch(cfg).expect("cuda dropout2d bwd: launch failed"); }
            accumulate_into(&in_bwd, &Rc::new(RefCell::new(temp)), total);
        }));
    }

    out_node
}