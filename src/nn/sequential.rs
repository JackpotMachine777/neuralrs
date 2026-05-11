use crate::tensor::Tensor;
 use crate::nn::module::Module;

pub struct Sequential{
    pub list: Vec<Box<dyn Module>>
}

impl Module for Sequential{
    fn forward(&mut self, input: &Tensor) -> Tensor{
        let mut current = input.clone();

        for layer in &mut self.list{
            current = layer.forward(&current);
        }

        current
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor{
        let mut grad = grad_output.clone();

        for layer in self.list.iter_mut().rev(){
            grad = layer.backward(&grad);
        }

        grad
    }
}