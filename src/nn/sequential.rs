use crate::tensor::Tensor;
use crate::nn::module::Module;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

pub struct Sequential{
    pub list: Vec<Box<dyn Module>>
}

impl Module for Sequential{
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let mut current = input;

        for layer in &mut self.list{
            current = layer.forward(current);
        }

        current
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![];

        for layer in &mut self.list{
            params.extend(layer.parameters());
        }

        params
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.list{
            layer.zero_grad();
        }
    }

    fn sync_grads(&mut self) {
        for layer in &mut self.list {
            layer.sync_grads();
        }
    }

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.list {
            layer.set_training(training);
        }
    }
}