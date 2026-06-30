pub mod cross_entropy;  pub use cross_entropy::{cross_entropy, cross_entropy_backward};
pub mod mse;            pub use mse::{mse, mse_backward};
pub mod bce;            pub use bce::{bce, bce_backward};