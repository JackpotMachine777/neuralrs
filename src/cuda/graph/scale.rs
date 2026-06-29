//! Scale by a fixed scalar (resident autograd op): out = x * s. Backward: grad * s.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vscale(float* out, const float* x, const float s, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = x[i] * s;
    }
        
    extern "C" __global__ void scale_grad(float* out, const float* g, const float s, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = g[i] * s;
    }
"#;
crate::kernel_module!(KERNEL);

/// Multiplies a resident node by the scalar `s`, kept on the GPU.
///
/// # Panics
/// If the input is not on the GPU.
pub fn scale(a: &Rc<RefCell<Node>>, s: f32) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda scale: input not on GPU");
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda scale: alloc failed");
        let f = module().load_function("vscale").expect("vscale not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&a_gpu.data); bd.arg(&s); bd.arg(&len);
        unsafe { bd.launch(cfg).expect("cuda scale: launch failed"); }
        (out, a_n.shape.clone(), len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda scale: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let mut local = stream.alloc_zeros::<f32>(len).expect("cuda scale bwd: alloc");
            let f = module().load_function("scale_grad").expect("scale_grad not found");
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut bd = stream.launch_builder(&f);
            bd.arg(&mut local); bd.arg(&*g); bd.arg(&s); bd.arg(&len);
            unsafe { bd.launch(cfg).expect("cuda scale bwd: launch"); }
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(local)), len);
        }));
    }
    out_node
}