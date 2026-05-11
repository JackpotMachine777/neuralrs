use crate::tensor::Tensor;
use crate::ops::matmul::matmul;
use crate::nn::module::Module;

pub struct Linear{
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Module for Linear{
    fn forward(&self, input: &Tensor) -> Tensor{
        if input.shape[1] != self.weights.shape[0]{
            panic!("Inputs are not matching weights");
        }

        let a = matmul(input, &self.weights);

        if self.bias.shape[0] != a.shape[1]{
            panic!("Bias is not matching weights shapes");
        }
        
        a.add(&self.bias)
    }
}