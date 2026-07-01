//! Neural-network layers on the GPU, operations that
//! build on the graph primitives: convolution, pooling, normalization, dropout.

pub mod batchnorm;      pub use batchnorm::batchnorm;
pub mod conv;           pub use conv::conv2d;
pub mod dropout;        pub use dropout::dropout;
pub mod flatten;        pub use flatten::flatten;
pub mod maxpool;        pub use maxpool::maxpool2d;
pub mod avgpool;        pub use avgpool::avgpool2d;
pub mod normalization;  pub use normalization::{layernorm, LayerNorm};
pub mod embedding;      pub use embedding::Embedding;
pub mod attention;      pub use attention::{attention, attention_batch};
pub mod rnn;            pub use rnn::RNNCell;
pub mod lstm;           pub use lstm::LSTMCell;
pub mod linear;         pub use linear::Linear;
pub mod positional;     pub use positional::PositionalEncoding;
pub mod self_attention; pub use self_attention::SelfAttention;
pub mod multihead;      pub use multihead::MultiHeadAttention;
pub mod transformer_block; pub use transformer_block::TransformerBlock;