//! Elementwise power (resident autograd op). Backward: grad * alpha * x^(alpha - 1).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vpow(float* out, const float* x, const float p, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = powf(x[i], p);
    }

    extern "C" __global__ void pow_grad(float* out, const float* g, const float* x, const float p, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = g[i] * p * powf(x[i], p - 1.0f);
    }
"#;
crate::kernel_module!(KERNEL);

/// Raises a resident node elementwise to the power `alpha`, kept on the GPU.
pub fn pow(a: &Rc<RefCell<Node>>, alpha: f32) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda pow: input not on GPU");
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda pow: alloc failed");
        let f = module().load_function("vpow").expect("vpow not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&a_gpu.data); bd.arg(&alpha); bd.arg(&len);
        unsafe { bd.launch(cfg).expect("cuda pow: launch failed"); }
        (out, a_n.shape.clone(), len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda pow: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let local = {
                let a_n = a_bwd.borrow();
                let x = &a_n.gpu.as_ref().expect("cuda pow bwd: input not on GPU").data;
                let mut out = stream.alloc_zeros::<f32>(len).expect("cuda pow bwd: alloc");
                let f = module().load_function("pow_grad").expect("pow_grad not found");
                let cfg = LaunchConfig::for_num_elems(len as u32);
                let mut bd = stream.launch_builder(&f);
                bd.arg(&mut out); bd.arg(&*g); bd.arg(x); bd.arg(&alpha); bd.arg(&len);
                unsafe { bd.launch(cfg).expect("cuda pow bwd: launch"); }
                out
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(local)), len);
        }));
    }
    out_node
}