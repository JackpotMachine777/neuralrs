//! # NeuralRs
//!
//! A deep learning library written from scratch in Rust — its own autograd
//! engine, a full neural-network stack, and a working Transformer.
//!
//! The pieces fit together in layers:
//!
//! - [`tensor`] — the core n-dimensional data type
//! - [`autograd`] — the engine that records operations and computes gradients
//!   (start with [`autograd::node`] to see how backprop works here)
//! - [`nn`] — layers, activations, and losses, all built on the autograd graph
//!   (the usual entry point is [`nn::sequential::Sequential`])
//! - [`optim`] — optimizers and learning-rate schedulers
//! - [`data`] — a batching/shuffling data loader and MNIST reader
//!
//! See the `examples/` directory for full training loops (MNIST CNN, a
//! Transformer on a toy task, and a minimal XOR net).

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
pub mod dispatch;
pub mod init;
pub mod serialize;
pub mod data;
pub mod train;

#[cfg(feature = "cuda")]
pub mod cuda;