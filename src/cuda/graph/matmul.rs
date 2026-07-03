//! Matrix multiply on the GPU (resident autograd op): C = A @ B.
//!
//! `C[m,n] = A[m,k] @ B[k,n]`, row-major. Backward: `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`.
//! Rather than transposing into scratch buffers, the backward uses two kernel
//! variants that read one operand in transposed order (like a BLAS transa/transb
//! flag): `matmul_nt` (B transposed) and `matmul_tn` (A transposed).
//!
//! Kernels are register-blocked: each 16x16 thread block computes a 64x64
//! output tile, every thread a 4x4 patch accumulated in registers from
//! shared-memory slabs of A and B.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;
use crate::cuda::runtime::accumulate_into;

const KERNEL: &str = r#"
    // Tiled matmul: each 16x16 block cooperatively loads TILE-wide slabs of A
    // and B into shared memory, then every thread accumulates its output element
    // from the shared tiles. Global reads per output drop from 2*inner to
    // 2*inner/TILE. TILE must match the block_dim set in mm() on the Rust side.
    // Tiles are padded to TILE+1 columns so the transposed variants' column-order
    // reads don't hit shared-memory bank conflicts.
    #define TILE 16

    // Register-blocked forward kernel: a 16x16 block computes a BMxBN output
    // tile, each thread a TMxTN patch held in registers (16 FMAs per 8 shared
    // loads). BM/BN must match the matmul_nn grid set in mm().
    #define BM 64
    #define BN 64
    #define BK 16
    #define TM 4
    #define TN 4

    extern "C" __global__ void matmul_nn(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        __shared__ float As[BK][BM + 1];   // A tile stored transposed: As[k][m] (+1 pad: conflict-free stores)
        __shared__ float Bs[BK][BN];       // B tile: Bs[k][n]

        int block_row = blockIdx.y * BM;
        int block_col = blockIdx.x * BN;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255

        int a_col = tid % BK;   // k of the A element this thread loads
        int a_row = tid / BK;   // m of the A element (stepped by 16 below)
        int b_col = tid % BN;   // n of the B element this thread loads
        int b_row = tid / BN;   // k of the B element (stepped by 4 below)

        float acc[TM][TN] = {};
        float a_reg[TM];
        float b_reg[TN];

        for (int t = 0; t < inner; t += BK) {
            for (int p = 0; p < BM; p += 256 / BK) {
                int m = a_row + p;
                int gr = block_row + m;
                int gk = t + a_col;
                As[a_col][m] = (gr < rows && gk < inner) ? A[gr * inner + gk] : 0.0f;
            }
            for (int p = 0; p < BK; p += 256 / BN) {
                int k = b_row + p;
                int gk = t + k;
                int gc = block_col + b_col;
                Bs[k][b_col] = (gk < inner && gc < cols) ? B[gk * cols + gc] : 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < BK; ++k) {
                for (int i = 0; i < TM; ++i) a_reg[i] = As[k][threadIdx.y * TM + i];
                for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][threadIdx.x * TN + j];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }

        for (int i = 0; i < TM; ++i) {
            int gr = block_row + threadIdx.y * TM + i;
            if (gr >= rows) continue;
            for (int j = 0; j < TN; ++j) {
                int gc = block_col + threadIdx.x * TN + j;
                if (gc < cols) C[gr * cols + gc] = acc[i][j];
            }
        }
    }

    extern "C" __global__ void matmul_nt(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        __shared__ float As[BK][BM + 1];
        __shared__ float Bs[BK][BN + 1];

        int block_row = blockIdx.y * BM;
        int block_col = blockIdx.x * BN;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        int a_col = tid % BK;
        int a_row = tid / BK;
        int b_k   = tid % BK;
        int b_c   = tid / BK;

        float acc[TM][TN] = {};
        float a_reg[TM];
        float b_reg[TN];

        for (int t = 0; t < inner; t += BK) {
            for (int p = 0; p < BM; p += 256 / BK) {
                int m = a_row + p;
                int gr = block_row + m;
                int gk = t + a_col;
                As[a_col][m] = (gr < rows && gk < inner) ? A[gr * inner + gk] : 0.0f;
            }
            for (int p = 0; p < BN; p += 256 / BK) {
                int n = b_c + p;
                int gc = block_col + n;
                int gk = t + b_k;
                Bs[b_k][n] = (gc < cols && gk < inner) ? B[gc * inner + gk] : 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < BK; ++k) {
                for (int i = 0; i < TM; ++i) a_reg[i] = As[k][threadIdx.y * TM + i];
                for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][threadIdx.x * TN + j];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }

        for (int i = 0; i < TM; ++i) {
            int gr = block_row + threadIdx.y * TM + i;
            if (gr >= rows) continue;
            for (int j = 0; j < TN; ++j) {
                int gc = block_col + threadIdx.x * TN + j;
                if (gc < cols) C[gr * cols + gc] = acc[i][j];
            }
        }
    }

    extern "C" __global__ void matmul_tn(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        __shared__ float As[BK][BM + 1];
        __shared__ float Bs[BK][BN];

        int block_row = blockIdx.y * BM;
        int block_col = blockIdx.x * BN;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        int a_m   = tid % BM;
        int a_k   = tid / BM;
        int b_col = tid % BN;
        int b_row = tid / BN;

        float acc[TM][TN] = {};
        float a_reg[TM];
        float b_reg[TN];

        for (int t = 0; t < inner; t += BK) {
            for (int p = 0; p < BK; p += 256 / BM) {
                int k = a_k + p;
                int gk = t + k;
                int gm = block_row + a_m;
                As[k][a_m] = (gk < inner && gm < rows) ? A[gk * rows + gm] : 0.0f;
            }
            for (int p = 0; p < BK; p += 256 / BN) {
                int k = b_row + p;
                int gk = t + k;
                int gc = block_col + b_col;
                Bs[k][b_col] = (gk < inner && gc < cols) ? B[gk * cols + gc] : 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < BK; ++k) {
                for (int i = 0; i < TM; ++i) a_reg[i] = As[k][threadIdx.y * TM + i];
                for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][threadIdx.x * TN + j];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }

        for (int i = 0; i < TM; ++i) {
            int gr = block_row + threadIdx.y * TM + i;
            if (gr >= rows) continue;
            for (int j = 0; j < TN; ++j) {
                int gc = block_col + threadIdx.x * TN + j;
                if (gc < cols) C[gr * cols + gc] = acc[i][j];
            }
        }
    }

    // Split-K variant of matmul_nt for skinny outputs with a huge inner
    // dimension (the conv weight-gradient shape: [c_out, c_in*kh*kw] reduced
    // over N*out_h*out_w). blockIdx.z slices the inner dimension so enough
    // blocks exist to fill the GPU despite the tiny output tile count; each
    // slice accumulates its partial tile with atomicAdd, so C must be
    // zero-initialized. Loads are guarded with k_end, not inner, so slices
    // never overlap.
    extern "C" __global__ void matmul_nt_splitk(float* C, const float* A, const float* B, int rows, int cols, int inner) {
        __shared__ float As[BK][BM + 1];
        __shared__ float Bs[BK][BN + 1];

        int block_row = blockIdx.y * BM;
        int block_col = blockIdx.x * BN;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        int a_col = tid % BK;
        int a_row = tid / BK;
        int b_k   = tid % BK;
        int b_c   = tid / BK;

        int per = (inner + gridDim.z - 1) / gridDim.z;
        int k_begin = blockIdx.z * per;
        int k_end = min(k_begin + per, inner);

        float acc[TM][TN] = {};
        float a_reg[TM];
        float b_reg[TN];

        for (int t = k_begin; t < k_end; t += BK) {
            for (int p = 0; p < BM; p += 256 / BK) {
                int m = a_row + p;
                int gr = block_row + m;
                int gk = t + a_col;
                As[a_col][m] = (gr < rows && gk < k_end) ? A[gr * inner + gk] : 0.0f;
            }
            for (int p = 0; p < BN; p += 256 / BK) {
                int n = b_c + p;
                int gc = block_col + n;
                int gk = t + b_k;
                Bs[b_k][n] = (gc < cols && gk < k_end) ? B[gc * inner + gk] : 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < BK; ++k) {
                for (int i = 0; i < TM; ++i) a_reg[i] = As[k][threadIdx.y * TM + i];
                for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][threadIdx.x * TN + j];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a_reg[i] * b_reg[j];
            }
            __syncthreads();
        }

        for (int i = 0; i < TM; ++i) {
            int gr = block_row + threadIdx.y * TM + i;
            if (gr >= rows) continue;
            for (int j = 0; j < TN; ++j) {
                int gc = block_col + threadIdx.x * TN + j;
                if (gc < cols) atomicAdd(&C[gr * cols + gc], acc[i][j]);
            }
        }
    }
"#;

crate::kernel_module!(KERNEL);

/// Raw device-level GEMM on slices, shared with ops that lower to a matrix
/// multiply (conv2d's im2col path). `kernel` picks the variant by name.
pub(crate) fn mm(kernel: &str, a: &CudaSlice<f32>, b: &CudaSlice<f32>, rows: usize, cols: usize, inner: usize) -> CudaSlice<f32> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(rows * cols).expect("cuda matmul: alloc failed");

    let f = module().load_function(kernel).expect("matmul kernel not found");
    // All variants are register-blocked: a 16x16 thread block computes a 64x64
    // output tile, so the grid steps in 64s. Must match BM/BN in the kernels.
    let cfg = LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(64), (rows as u32).div_ceil(64), 1),
        block_dim: (16, 16, 1),
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

/// Split-K GEMM for skinny outputs with a huge inner dimension (the conv
/// weight-gradient shape). Slices `inner` across `grid.z` so enough blocks
/// exist to fill the GPU; the kernel accumulates partial tiles with atomicAdd
/// into the zero-initialized output.
pub(crate) fn mm_nt_splitk(a: &CudaSlice<f32>, b: &CudaSlice<f32>, rows: usize, cols: usize, inner: usize) -> CudaSlice<f32> {
    let stream = backend::stream();
    let mut out = stream.alloc_zeros::<f32>(rows * cols).expect("cuda matmul: alloc failed");

    let f = module().load_function("matmul_nt_splitk").expect("matmul kernel not found");
    let base = (cols as u32).div_ceil(64) * (rows as u32).div_ceil(64);
    let max_slices = (inner as u32).div_ceil(16).max(1);
    let grid_z = (128 / base).clamp(1, 64).min(max_slices);
    let cfg = LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(64), (rows as u32).div_ceil(64), grid_z),
        block_dim: (16, 16, 1),
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