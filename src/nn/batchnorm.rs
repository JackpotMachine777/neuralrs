use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct BatchNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub epsilon: f32,
    pub num_features: usize
}

impl Module for BatchNorm {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();
        let n = data.len() as f32;

        let mean = data.iter().sum::<f32>() / n;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        
        let out: Vec<f32> = data.iter().enumerate().map(|(i, &x)| {
            let x_norm = (x - mean) / (var + self.epsilon).sqrt();
            self.gamma.data[i % self.num_features] * x_norm + self.beta.data[i % self.num_features]
        }).collect();

        let shape = input.borrow().shape.clone();
        Node::new(out, shape)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![&mut self.gamma, &mut self.beta] }

    fn zero_grad(&mut self) { 
        self.gamma.grad = vec![0.0; self.gamma.data.len()];
        self.beta.grad = vec![0.0; self.beta.data.len()];
    }
}