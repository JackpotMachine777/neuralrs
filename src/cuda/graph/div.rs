//! Element-wise divide (resident autograd op): out = a / b.
//! Backward (quotient rule): grad/b to a, -grad*a/b² to b.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vdiv(float* out, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = a[i] / b[i];
    }

    extern "C" __global__ void div_ga(float* out, const float* g, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = g[i] / b[i];
    }
        
    extern "C" __global__ void div_gb(float* out, const float* g, const float* a, const float* b, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = -g[i] * a[i] / (b[i] * b[i]);
    }
"#;
crate::kernel_module!(KERNEL);

/// Element-wise `a / b` of two resident nodes, kept on the GPU.
///
/// # Panics
/// If either input is not on the GPU.
pub fn div(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda div: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda div: rhs not on GPU");
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda div: alloc failed");
        let f = module().load_function("vdiv").expect("vdiv not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&a_gpu.data); bd.arg(&b_gpu.data); bd.arg(&len);
        unsafe { bd.launch(cfg).expect("cuda div: launch failed"); }
        (out, a_n.shape.clone(), len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone(), b.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda div: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        let b_bwd = b.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let cfg = LaunchConfig::for_num_elems(len as u32);

            let ga = {
                let b_n = b_bwd.borrow();
                let b_data = &b_n.gpu.as_ref().expect("cuda div bwd: rhs not on GPU").data;
                let mut out = stream.alloc_zeros::<f32>(len).expect("cuda div bwd: alloc");
                let f = module().load_function("div_ga").expect("div_ga not found");
                let mut bd = stream.launch_builder(&f);
                bd.arg(&mut out); bd.arg(&*g); bd.arg(b_data); bd.arg(&len);
                unsafe { bd.launch(cfg).expect("cuda div bwd: ga launch"); }
                out
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(ga)), len);

            let gb = {
                let a_n = a_bwd.borrow();
                let b_n = b_bwd.borrow();
                let a_data = &a_n.gpu.as_ref().expect("cuda div bwd: lhs not on GPU").data;
                let b_data = &b_n.gpu.as_ref().expect("cuda div bwd: rhs not on GPU").data;
                let mut out = stream.alloc_zeros::<f32>(len).expect("cuda div bwd: alloc");
                let f = module().load_function("div_gb").expect("div_gb not found");
                let mut bd = stream.launch_builder(&f);
                bd.arg(&mut out); bd.arg(&*g); bd.arg(a_data); bd.arg(b_data); bd.arg(&len);
                unsafe { bd.launch(cfg).expect("cuda div bwd: gb launch"); }
                out
            };
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(gb)), len);
        }));
    }
    out_node
}