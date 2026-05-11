use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct ReLU;

impl Module for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut out = Vec::with_capacity(input.data.len());

        for &x in &input.data {
            if x < 0.0 {
                out.push(0.0);
            } else {
                out.push(x);
            }
        }

        Tensor {
            data: out,
            grad: vec![0.0; input.data.len()],
            shape: input.shape.clone(),
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor{
        grad_output.clone()
    }
}