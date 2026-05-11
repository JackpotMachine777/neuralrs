use crate::tensor::Tensor;
 use crate::nn::module::Module;

pub struct Sequential{
    pub list: Vec<Box<dyn Module>>
}

impl Module for Sequential{
    fn forward(&self, input: &Tensor) -> Tensor{
        let mut current = input.clone();

        for layer in &self.list{
            current = layer.forward(&current);
        }

        current
    }
}