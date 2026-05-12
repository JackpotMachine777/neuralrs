use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct ReLU{
    pub last_input: Option<Tensor>,
}

impl Module for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.last_input = Some(input.clone());
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
        let last = self.last_input.as_ref().unwrap();
        let mut out = Vec::with_capacity(last.data.len());
        

        for i in 0..last.data.len(){
            if last.data[i] != 0.0{ out.push(grad_output.data[i]); }
            else { out.push(0.0); }
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