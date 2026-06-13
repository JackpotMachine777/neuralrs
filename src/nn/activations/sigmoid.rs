use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::graph;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

pub struct Sigmoid{ }

impl Module for Sigmoid{
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>>{
        graph::sigmoid(input)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor>{
        vec![]
    }

    fn zero_grad(&mut self){ }
}