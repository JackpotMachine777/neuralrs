use crate::storage::Storage;
use crate::dtype::DType;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub storage: Storage,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}