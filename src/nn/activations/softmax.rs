use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct Softmax{
    pub last_output: Option<Tensor>,
}

impl Module for Softmax{
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut exps = Vec::with_capacity(input.data.len());
        let mut sum = 0.0;
        let mut out = Vec::with_capacity(exps.len());

        for &x in &input.data {
            exps.push(x.exp());
        }

        for i in 0..exps.len() {
            sum += exps[i];
        }

        for j in 0..exps.len() {
            out.push(exps[j] / sum);
        }

        let res = Tensor {
            data: out,
            grad: vec![0.0; input.data.len()],
            shape: input.shape.clone(),
        };

        self.last_output = Some(res.clone());

        res
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let s = self.last_output.as_ref().unwrap();
        let mut out = Vec::with_capacity(grad_output.data.len());

        for i in 0..grad_output.data.len(){
            out.push(s.data[i] * (1.0 - s.data[i]) * grad_output.data[i]);
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