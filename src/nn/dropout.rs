use crate::nn::module::Module;
use crate::tensor::Tensor;

pub struct Dropout{
    pub probability: f32,
    pub mask: Vec<f32>,
    pub training: bool,
}

impl Module for Dropout{
    fn forward(&mut self, input: &Tensor) -> Tensor {
        if self.training == false { return input.clone(); }

        let mut out = Vec::with_capacity(input.data.len());

        for &x in &input.data {
            let n = rand::random::<f32>();

            if n < self.probability { 
                self.mask.push(0.0);
                out.push(0.0); 
            }
            else { 
                self.mask.push(1.0);
                out.push(x * (1.0 / (1.0 - self.probability))); 
            }
        }

        Tensor {
            data: out,
            grad: vec![0.0; input.data.len()],
            shape: input.shape.clone(),
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut out = Vec::with_capacity(grad_output.data.len());

        for i in 0..grad_output.data.len(){
            out.push(grad_output.data[i] * self.mask[i]);
        }

        Tensor {
            data: out,
            grad: grad_output.grad.clone(),
            shape: grad_output.shape.clone(),
        }
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }

    fn zero_grad(&mut self) { }
}