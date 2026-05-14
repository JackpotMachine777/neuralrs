use crate::tensor::Tensor;
use crate::nn::module::Module;

use crate::ops::matmul::matmul;
use crate::ops::elementwise::arithmetic::add_vec;
use crate::ops::shape::transpose;

pub struct Linear{
    pub weights: Tensor,
    pub bias: Tensor,
    pub last_input: Option<Tensor>,
}

impl Module for Linear{
    fn forward(&mut self, input: &Tensor) -> Tensor{
        self.last_input = Some(input.clone());

        if input.shape[1] != self.weights.shape[0]{
            panic!("Inputs are not matching weights");
        }

        let a = matmul(input, &self.weights);

        if self.bias.shape[0] != a.shape[1]{
            panic!("Bias is not matching weights shapes");
        }
        
        a.add(&self.bias)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor{
        let input = self.last_input.as_ref().unwrap();
        let grad_out = grad_output;
        let x = input;
        let w = &self.weights;

        let grad_w = matmul(&transpose(x), grad_out);
        let mut grad_b = vec![0.0; self.bias.data.len()];

        let len = grad_b.len();

        for i in 0..grad_output.data.len(){
            grad_b[i % len] += grad_output.data[i];
        }

        let grad_input = matmul(grad_output, &transpose(w));

        self.weights.grad = add_vec(&self.weights.grad, &grad_w.data);
        self.bias.grad = add_vec(&self.bias.grad, &grad_b);

        grad_input
    }

    fn parameters(&mut self) -> Vec<&mut Tensor>{
        vec![&mut self.weights, &mut self.bias]
    }

    fn zero_grad(&mut self){
        self.weights.grad = vec![0.0; self.weights.data.len()];
        self.bias.grad = vec![0.0; self.bias.data.len()];
    }
}