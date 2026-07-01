//! Vanilla RNN cell on the GPU, composed from resident graph ops (matmul, add,
//! tanh). Weights are resident nodes; autograd flows through the composition.
//! Mirrors the CPU `RNNCell`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;

/// A vanilla RNN cell over resident weight nodes. `step` computes
/// `h_new = tanh(x @ w_xh + h_prev @ w_hh + bias)`.
pub struct RNNCell {
    pub w_xh: Rc<RefCell<Node>>,
    pub w_hh: Rc<RefCell<Node>>,
    pub bias: Rc<RefCell<Node>>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl RNNCell {
    /// Builds a cell from resident weight nodes: `w_xh [input, hidden]`,
    /// `w_hh [hidden, hidden]`, `bias [hidden]`.
    pub fn new(
        w_xh: Rc<RefCell<Node>>,
        w_hh: Rc<RefCell<Node>>,
        bias: Rc<RefCell<Node>>,
        input_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self { w_xh, w_hh, bias, input_size, hidden_size }
    }

    /// One timestep: `x [batch, input]`, `h_prev [batch, hidden]` -> `[batch, hidden]`.
    pub fn step(&self, x: &Rc<RefCell<Node>>, h_prev: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let xh = graph::matmul(x, &self.w_xh);
        let hh = graph::matmul(h_prev, &self.w_hh);
        let sum = graph::add(&xh, &hh);
        let with_bias = graph::add(&sum, &self.bias);
        graph::tanh(&with_bias)
    }
}