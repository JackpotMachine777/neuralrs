use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vsilu(float* out, const float* x, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) { float s = 1.0f / (1.0f + expf(-x[i])); out[i] = x[i] * s; }
    }

    extern "C" __global__ void silu_grad(float* out, const float* g, const float* x, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) { float s = 1.0f / (1.0f + expf(-x[i])); out[i] = g[i] * (s + x[i] * s * (1.0f - s)); }
    }
"#;
crate::kernel_module!(KERNEL);

pub fn silu(a: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda silu: input not on GPU");
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda silu: alloc failed");
        let f = module().load_function("vsilu").expect("vsilu not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&a_gpu.data); bd.arg(&len);
        unsafe { bd.launch(cfg).expect("cuda silu: launch failed"); }
        (out, a_n.shape.clone(), len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda silu: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let local = {
                let a_n = a_bwd.borrow();
                let x = &a_n.gpu.as_ref().expect("cuda silu bwd: input not on GPU").data;
                let mut out = stream.alloc_zeros::<f32>(len).expect("cuda silu bwd: alloc");
                let f = module().load_function("silu_grad").expect("silu_grad not found");
                let cfg = LaunchConfig::for_num_elems(len as u32);
                let mut bd = stream.launch_builder(&f);
                bd.arg(&mut out); bd.arg(&*g); bd.arg(x); bd.arg(&len);
                unsafe { bd.launch(cfg).expect("cuda silu bwd: launch"); }
                out
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(local)), len);
        }));
    }
    out_node
}