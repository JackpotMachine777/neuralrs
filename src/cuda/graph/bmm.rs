//! Batched matmul (resident autograd op): [batch, m, k] x [batch, k, n] ->
//! [batch, m, n]. Per-batch matmul; backward applies matmul grad rules per item.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void bmm_forward(float* out, const float* a, const float* b, int batch, int m, int k, int n) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * m * n;
        if(t < total) {
            int bi = t / (m * n);
            int rem = t % (m * n);
            int i = rem / n;
            int j = rem % n;
            int a_off = bi * m * k;
            int b_off = bi * k * n;
            float sum = 0.0f;
            for(int p = 0; p < k; ++p) sum += a[a_off + i * k + p] * b[b_off + p * n + j];
            out[bi * m * n + i * n + j] = sum;
        }
    }
    extern "C" __global__ void bmm_grad_a(float* da, const float* grad, const float* b, int batch, int m, int k, int n) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * m * k;
        if(t < total) {
            int bi = t / (m * k);
            int rem = t % (m * k);
            int i = rem / k;
            int p = rem % k;
            int o_off = bi * m * n;
            int b_off = bi * k * n;
            float g = 0.0f;
            for(int j = 0; j < n; ++j) g += grad[o_off + i * n + j] * b[b_off + p * n + j];
            da[bi * m * k + i * k + p] = g;
        }
    }
    extern "C" __global__ void bmm_grad_b(float* db, const float* a, const float* grad, int batch, int m, int k, int n) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch * k * n;
        if(t < total) {
            int bi = t / (k * n);
            int rem = t % (k * n);
            int p = rem / n;
            int j = rem % n;
            int o_off = bi * m * n;
            int a_off = bi * m * k;
            float g = 0.0f;
            for(int i = 0; i < m; ++i) g += a[a_off + i * k + p] * grad[o_off + i * n + j];
            db[bi * k * n + p * n + j] = g;
        }
    }
"#;
crate::kernel_module!(KERNEL);

/// Batched matmul of two resident nodes, [batch, m, k] x [batch, k, n] ->
/// [batch, m, n], kept on the GPU.
///
/// # Panics
/// If either input is not on the GPU.
pub fn bmm(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();
    let (out_data, batch, m, k, n) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda bmm: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda bmm: rhs not on GPU");
        let batch = a_n.shape[0];
        let m = a_n.shape[1];
        let k = a_n.shape[2];
        let n = b_n.shape[2];
        let out_len = batch * m * n;

        let mut out = stream.alloc_zeros::<f32>(out_len).expect("cuda bmm: alloc failed");
        let f = module().load_function("bmm_forward").expect("bmm_forward not found");
        let (bb, mm, kk, nn) = (batch as i32, m as i32, k as i32, n as i32);
        let cfg = LaunchConfig::for_num_elems(out_len as u32);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut out); builder.arg(&a_gpu.data); builder.arg(&b_gpu.data);
        builder.arg(&bb); builder.arg(&mm); builder.arg(&kk); builder.arg(&nn);
        unsafe { builder.launch(cfg).expect("cuda bmm: launch failed"); }
        (out, batch, m, k, n)
    };
    let out_node = Node::new(vec![], vec![batch, m, n]);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];
        let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(batch * m * n).expect("cuda bmm: grad alloc")));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });
        let a_bwd = a.clone();
        let b_bwd = b.clone();
        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let stream = backend::stream();
            let g = grad.borrow();
            let (bb, mm, kk, nn) = (batch as i32, m as i32, k as i32, n as i32);

            let da = {
                let b_n = b_bwd.borrow();
                let bd = &b_n.gpu.as_ref().expect("cuda bmm bwd: rhs not on GPU").data;
                let mut da = stream.alloc_zeros::<f32>(batch * m * k).expect("cuda bmm bwd: da alloc");
                let f = module().load_function("bmm_grad_a").expect("bmm_grad_a not found");
                let cfg = LaunchConfig::for_num_elems((batch * m * k) as u32);
                let mut builder = stream.launch_builder(&f);
                builder.arg(&mut da); builder.arg(&*g); builder.arg(bd);
                builder.arg(&bb); builder.arg(&mm); builder.arg(&kk); builder.arg(&nn);
                unsafe { builder.launch(cfg).expect("cuda bmm bwd: grad_a launch"); }
                da
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(da)), batch * m * k);

            let db = {
                let a_n = a_bwd.borrow();
                let ad = &a_n.gpu.as_ref().expect("cuda bmm bwd: lhs not on GPU").data;
                let mut db = stream.alloc_zeros::<f32>(batch * k * n).expect("cuda bmm bwd: db alloc");
                let f = module().load_function("bmm_grad_b").expect("bmm_grad_b not found");
                let cfg = LaunchConfig::for_num_elems((batch * k * n) as u32);
                let mut builder = stream.launch_builder(&f);
                builder.arg(&mut db); builder.arg(ad); builder.arg(&*g);
                builder.arg(&bb); builder.arg(&mm); builder.arg(&kk); builder.arg(&nn);
                unsafe { builder.launch(cfg).expect("cuda bmm bwd: grad_b launch"); }
                db
            };
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(db)), batch * k * n);
        }));
    }
    out_node
}