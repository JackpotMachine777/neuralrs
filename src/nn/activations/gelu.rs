use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::graph;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

/// GELU as a `Module`, wrapping the `gelu` autograd op.
pub struct GELU {}

impl Module for GELU {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        graph::gelu(input)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }

    fn zero_grad(&mut self) {}
}