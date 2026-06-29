use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void vgelu(float* out, const float* x, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            float xi = x[i];
            float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
            out[i] = 0.5f * xi * (1.0f + tanhf(inner));
        }
    }

    extern "C" __global__ void gelu_grad(float* out, const float* g, const float* x, const size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) {
            float xi = x[i];
            float c = 0.7978845608f;
            float inner = c * (xi + 0.044715f * xi * xi * xi);
            float t = tanhf(inner);
            float dinner = c * (1.0f + 3.0f * 0.044715f * xi * xi);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * xi * (1.0f - t * t) * dinner;
            out[i] = g[i] * dgelu;
        }
    }
"#;
crate::kernel_module!(KERNEL);

pub fn gelu(a: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, shape, len) = {
        let a_n = a.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda gelu: input not on GPU");
        let len: usize = a_n.shape.iter().product();
        let mut out = stream.alloc_zeros::<f32>(len).expect("cuda gelu: alloc failed");
        let f = module().load_function("vgelu").expect("vgelu not found");
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut bd = stream.launch_builder(&f);
        bd.arg(&mut out); bd.arg(&a_gpu.data); bd.arg(&len);
        unsafe { bd.launch(cfg).expect("cuda gelu: launch failed"); }
        (out, a_n.shape.clone(), len)
    };
    let out_node = Node::new(vec![], shape);
    {
        let mut n = out_node.borrow_mut();
        n.parents = vec![a.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(len).expect("cuda gelu: grad alloc")));
        n.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        n.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let local = {
                let a_n = a_bwd.borrow();
                let x = &a_n.gpu.as_ref().expect("cuda gelu bwd: input not on GPU").data;
                let mut out = stream.alloc_zeros::<f32>(len).expect("cuda gelu bwd: alloc");
                let f = module().load_function("gelu_grad").expect("gelu_grad not found");
                let cfg = LaunchConfig::for_num_elems(len as u32);
                let mut bd = stream.launch_builder(&f);
                bd.arg(&mut out); bd.arg(&*g); bd.arg(x); bd.arg(&len);
                unsafe { bd.launch(cfg).expect("cuda gelu bwd: launch"); }
                out
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(local)), len);
        }));
    }
    out_node
}