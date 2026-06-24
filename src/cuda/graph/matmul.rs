//! Matrix multiply on the GPU (resident autograd op): C = A @ B.
//!
//! C[m,n] = A[m,k] @ B[k,n], row-major. Backward: dA = dC @ Bᵀ, dB = Aᵀ @ dC.
//! Rather than transposing into scratch buffers, the backward uses two kernel
//! variants that read one operand in transposed order (like a BLAS transa/transb
//! flag): `matmul_nt` (B transposed) and `matmul_tn` (A transposed).
//!
//! Naive one-thread-per-output kernels for now, tiling is a later perf pass.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    extern "C" __global__ void matmul_nn(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;

        if(r < rows && c < cols) {
            float acc = 0.0f;
            for(int i = 0; i < inner; ++i) 
                acc += A[r * inner + i] * B[i * cols + c];

            C[r * cols + c] = acc;
        }
    }

    extern "C" __global__ void matmul_nt(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;

        if(r < rows && c < cols) {
            float acc = 0.0f;
            for(int i = 0; i < inner; ++i)
                acc += A[r * inner + i] * B[c * inner + i];

            C[r * cols + c] = acc;
        }
    }

    extern "C" __global__ void matmul_tn(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;

        if (r < rows && c < cols) {
            float acc = 0.0f;
            for (int i = 0; i < inner; ++i) 
                acc += A[i * rows + r] * B[i * cols + c];
            
            C[r * cols + c] = acc;
        }
    }
"#;

crate::kernel_module!(KERNEL);

fn mm(kernel: &str, a: &CudaSlice<f32>, b: &CudaSlice<f32>, rows: usize, cols: usize, inner: usize) -> CudaSlice<f32> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(rows * cols).expect("cuda matmul: alloc failed");

    const TILE: u32 = 16;
    let f = module().load_function(kernel).expect("matmul kernel not found");
    let cfg = LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(TILE), (rows as u32).div_ceil(TILE), 1),
        block_dim: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    };

    let (r, c, i) = (rows as i32, cols as i32, inner as i32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut out);
    builder.arg(a);
    builder.arg(b);
    builder.arg(&r);
    builder.arg(&c);
    builder.arg(&i);
    unsafe { builder.launch(cfg).expect("cuda matmul: launch failed"); }

    out
}

/// Matrix product of two resident nodes: `C = A @ B`, computed and kept on the
/// GPU. `A` is `[m,k]`, `B` is `[k,n]`, result is `[m,n]`.
///
/// # Panics
/// If either input is not on the GPU, or the inner dimensions don't match.
pub fn matmul(a: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let stream = backend::stream();

    let (out_data, m, k, n) = {
        let a_n = a.borrow();
        let b_n = b.borrow();
        let a_gpu = a_n.gpu.as_ref().expect("cuda matmul: lhs not on GPU");
        let b_gpu = b_n.gpu.as_ref().expect("cuda matmul: rhs not on GPU");

        let m = a_n.shape[0];
        let k = a_n.shape[1];
        let n = b_n.shape[1];
        assert_eq!(b_n.shape[0], k, "cuda matmul: inner dims mismatch");

        let out = mm("matmul_nn", &a_gpu.data, &b_gpu.data, m, n, k);
        (out, m, k, n)
    };

    let out_node = Node::new(vec![], vec![m, n]);
    {
        let mut node = out_node.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];

        let grad = Rc::new(RefCell::new(
            stream.alloc_zeros::<f32>(m * n).expect("cuda matmul: grad alloc failed"),
        ));
        node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

        let a_bwd = a.clone();
        let b_bwd = b.clone();
        node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
            let dc = grad.borrow();

            let da = {
                let b_n = b_bwd.borrow();
                let b_data = &b_n.gpu.as_ref().expect("cuda matmul bwd: rhs not on GPU").data;
                mm("matmul_nt", &dc, b_data, m, k, n)
            };
            accumulate_into(&a_bwd, &Rc::new(RefCell::new(da)), m * k);

            let db = {
                let a_n = a_bwd.borrow();
                let a_data = &a_n.gpu.as_ref().expect("cuda matmul bwd: lhs not on GPU").data;
                mm("matmul_tn", a_data, &dc, k, n, m)
            };
            accumulate_into(&b_bwd, &Rc::new(RefCell::new(db)), k * n);
        }));
    }

    out_node
}