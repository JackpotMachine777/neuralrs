use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
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
            shape: input.shape.clone(),
        }
    }
}