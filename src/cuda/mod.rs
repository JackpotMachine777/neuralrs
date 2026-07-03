//! CUDA backend, opt-in GPU acceleration (`--features cuda`).
//!
//! Kernels are written in CUDA C inside each op file and compiled at runtime
//! with NVRTC (no nvcc step, no `.cu` files). The shared device context and
//! stream live in [`backend`]; each op compiles and caches its own kernels.

mod backend;
mod reduce;

pub mod graph;
pub mod nn;
pub mod runtime;
pub mod loss;
pub mod optim;