use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::graph;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

/// Tanh as a `Module`, wrapping the `tanh` autograd op.
pub struct Tanh{ }

impl Module for Tanh{
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>>{
        graph::tanh(input)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor>{
        vec![]
    }

    fn zero_grad(&mut self){ }
}