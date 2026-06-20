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