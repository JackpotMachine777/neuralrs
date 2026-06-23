//! CUDA backend, opt-in GPU acceleration (`--features cuda`).
//!
//! Kernels are written in CUDA C inside each op file and compiled at runtime
//! with NVRTC (no nvcc step, no `.cu` files). The shared device context and
//! stream live in [`backend`]; each op compiles and caches its own kernels.

mod backend;

pub mod add;            pub use add::add;
pub mod mul;            pub use mul::mul;
pub mod matmul;         pub use matmul::{matmul, matmul_naive};
pub mod graph;