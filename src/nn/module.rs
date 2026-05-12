use crate::tensor::Tensor;

pub trait Module {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
    fn zero_grad(&mut self);
}