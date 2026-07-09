//! # NeuralRs
//!
//! A deep learning library written from scratch in Rust, with its own autograd
//! engine, a full neural-network stack, and a working Transformer.
//!
//! The pieces fit together in layers:
//!
//! - [`tensor`] - the core n-dimensional data type
//! - [`autograd`] - the engine that records operations and computes gradients
//!   (start with [`autograd::node`] to see how backprop works here)
//! - [`nn`] - layers, activations, and losses, all built on the autograd graph
//!   (the usual entry point is [`nn::sequential::Sequential`])
//! - [`optim`] - optimizers and learning-rate schedulers
//! - [`data`] - a batching/shuffling data loader with MNIST and CIFAR-10 readers
//! - `cuda` (with `--features cuda`) - an optional resident GPU backend:
//!   every op, layer, and optimizer above has a device twin, each
//!   gradient-checked against the CPU implementation
//!
//! See the `examples/` directory for full training loops (MNIST CNN, with a
//! fully GPU-resident variant under `--features cuda`, a CIFAR-10 CNN, a
//! Transformer on a toy task, and a minimal XOR net), and `GUIDE.md` for a
//! walkthrough of building and training your own model.
//!
//! Checkpoints save through [`serialize::save`] and [`serialize::load`], which
//! pick the format by file extension: a human-readable positional text format
//! (zero dependencies, git-diffable, no names or shapes) or `.safetensors`
//! (binary, carries names, shapes, and dtype, and opens directly in PyTorch).

// These clippy lints flag stylistic choices that are intentional in this codebase.
// The autograd engine indexes parallel buffers (grad[i], data[i]) by hand for clarity,
// and the matmul/conv kernels genuinely need many stride arguments.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_div_ceil)]

pub mod dtype;
pub mod device;
pub mod storage;
pub mod tensor;
pub mod tensor_impl;
pub mod ops;
pub mod autograd;
pub mod nn;
pub mod optim;
pub mod init;
pub mod serialize;
pub mod data;
pub mod train;

#[cfg(feature = "cuda")]
pub mod cuda;