//! Element-wise subtract (resident autograd op): out = a - b.
//! Backward: +grad to a, -grad to b (via the shared accumulate).

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vsub(float* out, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = a[i] - b[i];
    }
        
    extern "C" __global__ void neg(float* out, const float* g, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = -g[i];
    }
"#;
crate::kernel_module!(KERNEL);

/// Element-wise `a - b` of two resident nodes, kept on the GPU.
///
/// # Panics
/// If either input is not on the GPU.
pub fn sub(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda sub: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda sub: rhs not on GPU");
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda sub: alloc failed");
        let f = module().load_function("vsub").expect("vsub not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&a_gpu.data); bd.arg(&b_gpu.data); bd.arg(&len);
        unsafe { bd.launch(cfg).expect("cuda sub: launch failed"); }
        (out, a_n.shape.clone(), len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone(), b.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda sub: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        let b_bwd = b.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            accumulate_into(&a_bwd, &grad, len);
            let neg = {
                let stream = backend::stream();
                let g = grad.borrow();
                let mut out = stream.alloc_zeros::<f32>(len).expect("cuda sub bwd: alloc");
                let f = module().load_function("neg").expect("neg not found");
                let cfg = LaunchConfig::for_num_elems(len as u32);
                let mut bd = stream.launch_builder(&f);
                bd.arg(&mut out); bd.arg(&*g); bd.arg(&len);
                unsafe { bd.launch(cfg).expect("cuda sub bwd: launch"); }
                out
            };
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(neg)), len);
        }));
    }
    out_node
}