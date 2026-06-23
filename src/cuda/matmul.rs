use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaModule, LaunchConfig, PushKernelArg};

use super::backend;

const TILE: u32 = 16;

const KERNEL: &str = r#"
    #define TILE 16

    extern "C" __global__ void  matmul_naive(float* C, const float* A, const float* B, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < M && col < N) {
            float acc = 0.0f;
            
            for(int k = 0; k < K; ++k) {
                acc += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = acc;
        }
    }

    extern "C" __global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        float acc = 0.0f;
        int num_tiles = (K + TILE - 1) / TILE;

        for(int t = 0; t < num_tiles; ++t) {
            int a_col = t * TILE + threadIdx.x;
            int b_row = t * TILE + threadIdx.y;

            As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
            
            __syncthreads();
            for(int k = 0; k < TILE; ++k) 
                acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
            __syncthreads();
        }

        if(row < M && col < N) C[row * N + col] = acc;
    }
"#;

static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();

fn module() -> &'static Arc<CudaModule> {
    MODULE.get_or_init(|| backend::compile(KERNEL))
}

fn launch(kernel: &str, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "cuda matmul: A must have M*K elements");
    assert_eq!(b.len(), k * n, "cuda matmul: B must have K*N elements");

    let stream = backend::stream();
    let a_dev = stream.clone_htod(a).expect("htod failed (A)");
    let b_dev = stream.clone_htod(b).expect("htod failed (B)");
    let mut c_dev = stream.alloc_zeros::<f32>(m * n).expect("device alloc failed (C)");

    let func = module().load_function(kernel).expect("kernel not found");
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(TILE), (m as u32).div_ceil(TILE), 1),
        block_dim: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    };

    let (mm, nn, kk) = (m as i32, n as i32, k as i32);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut c_dev);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mm);
    builder.arg(&nn);
    builder.arg(&kk);
    unsafe { builder.launch(cfg).expect("kernel launch failed"); }

    stream.clone_dtoh(&c_dev).expect("dtoh failed (C)")
}

/// Tiled matmul (shared memory). The canonical GPU matmul.
///
/// `C[m,n] = A[m,k] * B[k,n]`, all row-major. Inputs are copied to the device,
/// the kernel runs, and the result is copied back.
///
/// # Panics
/// If `a.len() != m*k` or `b.len() != k*n`.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    launch("matmul_tiled", a, b, m, k, n)
}

/// Naive matmul (one thread per output element, no shared memory). Same result
/// as [`matmul`], kept for benchmarking the two against each other.
pub fn matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    launch("matmul_naive", a, b, m, k, n)
}
