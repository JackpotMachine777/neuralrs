use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct Sigmoid{
    pub last_input: Option<Tensor>,
}

impl Module for Sigmoid{
    fn forward(&mut self, input: &Tensor) -> Tensor{
        self.last_input = Some(input.clone());
        let mut out = Vec::with_capacity(input.data.len());

        for &x in &input.data{
            out.push(1.0 / (1.0 + (-x).exp()));
        }

        Tensor{
            data: out,
            grad: vec![0.0; input.data.len()],
            shape: input.shape.clone(),
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor{
        let last = self.last_input.as_ref().unwrap();
        let mut out = Vec::with_capacity(last.data.len());

        for i in 0..last.data.len(){
            let s = 1.0 / (1.0 + (-last.data[i]).exp());
            out.push(s * (1.0 - s) * grad_output.data[i]);
        }

        Tensor{
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