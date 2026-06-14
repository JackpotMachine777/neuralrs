use crate::tensor::Tensor;
use crate::nn::module::Module;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;
use crate::autograd::graph;

pub struct Linear{
    pub weights: Tensor,
    pub bias: Tensor,
    pub weights_node: Option<Rc<RefCell<Node>>>,
    pub bias_node: Option<Rc<RefCell<Node>>>,
}

impl Module for Linear{
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>>{
        let w = Node::new(self.weights.storage.data.clone(), self.weights.shape.clone());
        let b = Node::new(self.bias.storage.data.clone(), self.bias.shape.clone());
        self.weights_node = Some(w.clone());
        self.bias_node = Some(b.clone());

        let a = graph::matmul(input, w);
        graph::add(a, b)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor>{
        vec![&mut self.weights, &mut self.bias]
    }

    fn zero_grad(&mut self){
        self.weights.grad = vec![0.0; self.weights.storage.data.len()];
        self.bias.grad = vec![0.0; self.bias.storage.data.len()];
    }

    fn sync_grads(&mut self) {
        if let Some(w) = &self.weights_node {
            self.weights.grad = w.borrow().grad.clone();
        }

        if let Some(b) = &self.bias_node {
            self.bias.grad = b.borrow().grad.clone();
        }
    }
}