//! Fully-connected layer on the GPU: `output = input @ weights + bias`, composed
//! from resident matmul + bias-add. Mirrors the CPU `Linear`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;

/// A fully-connected layer over resident weight/bias nodes: `weights [in, out]`,
/// `bias [out]`.
pub struct Linear {
    pub weights: Rc<RefCell<Node>>,
    pub bias: Rc<RefCell<Node>>,
}

impl Linear {
    pub fn new(weights: Rc<RefCell<Node>>, bias: Rc<RefCell<Node>>) -> Self {
        Self { weights, bias }
    }

    /// `input [batch, in] -> [batch, out]`.
    pub fn forward(&self, input: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let a = graph::matmul(input, &self.weights);
        graph::add(&a, &self.bias)
    }
}