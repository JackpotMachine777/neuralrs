//! Single-head self-attention on the GPU. Projects the input into Q/K/V, then
//! attends. Composed from resident matmul + attention. Mirrors the CPU
//! `SelfAttention`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;
use crate::cuda::nn::attention::attention;

/// Single-head self-attention over resident projection weights `[d_model, d_k]`.
pub struct SelfAttention {
    pub w_q: Rc<RefCell<Node>>,
    pub w_k: Rc<RefCell<Node>>,
    pub w_v: Rc<RefCell<Node>>,
    pub d_model: usize,
    pub d_k: usize,
}

impl SelfAttention {
    pub fn new(
        w_q: Rc<RefCell<Node>>,
        w_k: Rc<RefCell<Node>>,
        w_v: Rc<RefCell<Node>>,
        d_model: usize,
        d_k: usize,
    ) -> Self {
        Self { w_q, w_k, w_v, d_model, d_k }
    }

    /// `x [seq, d_model] -> [seq, d_k]`.
    pub fn forward(&self, x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let q = graph::matmul(x, &self.w_q);
        let k = graph::matmul(x, &self.w_k);
        let v = graph::matmul(x, &self.w_v);
        attention(&q, &k, &v)
    }
}