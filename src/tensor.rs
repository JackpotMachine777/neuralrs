use crate::storage::Storage;
use crate::dtype::DType;

/// The main data type — an n-dimensional array of `f32`.
///
/// Under the hood it's just a flat list of numbers plus a `shape` that says how
/// to read them: six values with shape `[2, 3]` means a 2×3 grid, even though in
/// memory it's one straight run. This is how PyTorch does it too — one
/// contiguous buffer, any number of dimensions.
///
/// Every tensor also carries a `grad` buffer the same length as its data (one
/// gradient slot per value), filled during backprop, plus its element type
/// (`dtype`, always `Float32` here).
///
/// Heads up: the methods on `Tensor` itself ([`Tensor::add`], [`Tensor::mul`])
/// are plain number-crunching and **don't** build an autograd graph. The
/// differentiable versions live in the autograd graph module.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub storage: Storage,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}