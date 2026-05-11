use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
}