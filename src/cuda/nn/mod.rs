//! Neural-network layers on the GPU, operations that
//! build on the graph primitives: convolution, pooling, normalization, dropout.

pub mod batchnorm;      pub use batchnorm::batchnorm;
pub mod conv;           pub use conv::conv2d;
pub mod dropout;        pub use dropout::dropout;
pub mod flatten;        pub use flatten::flatten;
pub mod maxpool;        pub use maxpool::maxpool2d;