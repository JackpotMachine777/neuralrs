use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct Tanh{
    pub last_input: Option<Tensor>,
}

impl Module for Tanh{
    fn forward(&mut self, input: &Tensor) -> Tensor{
        self.last_input = Some(input.clone());
        let mut out = Vec::with_capacity(input.data.len());

        for i in 0..input.data.len(){
            out.push(input.data[i].tanh());
        }

        Tensor {
            data: out,
            grad: vec![0.0; input.data.len()],
            shape: input.shape.clone(),
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor{
        let last = self.last_input.as_ref().unwrap();
        let mut out = Vec::with_capacity(last.data.len());

        for i in 0..last.data.len(){
            out.push((1.0 - last.data[i].tanh() * last.data[i].tanh()) * grad_output.data[i]);
        }

        Tensor {
            data: out,
            grad: grad_output.grad.clone(),
            shape: grad_output.shape.clone(),
        }
    }

    fn parameters(&mut self) -> Vec<&mut Tensor>{
        vec![]
    }

    fn zero_grad(&mut self){ }
}