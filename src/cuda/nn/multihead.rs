//! Multi-head self-attention on the GPU. Splits the model dim into heads, runs
//! attention per head, concatenates, and projects out. Composed from resident
//! matmul + slice_cols + attention + concat_cols (+ reshape for the batched
//! path). Mirrors the CPU `MultiHeadAttention`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;
use crate::cuda::nn::attention::{attention, attention_batch};

/// Multi-head self-attention over resident weights `w_q/w_k/w_v/w_o
/// [d_model, d_model]`. `d_model` must be divisible by `num_heads`.
pub struct MultiHeadAttention {
    pub w_q: Rc<RefCell<Node>>,
    pub w_k: Rc<RefCell<Node>>,
    pub w_v: Rc<RefCell<Node>>,
    pub w_o: Rc<RefCell<Node>>,
    pub d_model: usize,
    pub num_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(
        w_q: Rc<RefCell<Node>>,
        w_k: Rc<RefCell<Node>>,
        w_v: Rc<RefCell<Node>>,
        w_o: Rc<RefCell<Node>>,
        d_model: usize,
        num_heads: usize,
    ) -> Self {
        Self { w_q, w_k, w_v, w_o, d_model, num_heads }
    }

    /// Single-sequence: `x [seq, d_model] -> [seq, d_model]`.
    pub fn forward(&self, x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let q = graph::matmul(x, &self.w_q);
        let k = graph::matmul(x, &self.w_k);
        let v = graph::matmul(x, &self.w_v);

        let d_head = self.d_model / self.num_heads;
        let mut heads = Vec::new();
        for h in 0..self.num_heads {
            let start = h * d_head;
            let end = start + d_head;
            let q_h = graph::slice_cols(&q, start, end);
            let k_h = graph::slice_cols(&k, start, end);
            let v_h = graph::slice_cols(&v, start, end);
            heads.push(attention(&q_h, &k_h, &v_h));
        }

        let concat = graph::concat_cols(&heads);
        graph::matmul(&concat, &self.w_o)
    }

    /// Batched: `x [batch, seq, d_model] -> [batch, seq, d_model]`.
    pub fn forward_batch(&self, x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let (batch, seq, d) = { let s = x.borrow(); (s.shape[0], s.shape[1], s.shape[2]) };

        let proj = |x: &Rc<RefCell<Node>>, w: &Rc<RefCell<Node>>| -> Rc<RefCell<Node>> {
            let flat = graph::reshape(x, vec![batch * seq, d]);
            let out = graph::matmul(&flat, w);
            graph::reshape(&out, vec![batch, seq, d])
        };

        let q = proj(x, &self.w_q);
        let k = proj(x, &self.w_k);
        let v = proj(x, &self.w_v);

        let d_head = self.d_model / self.num_heads;
        let mut heads = Vec::new();
        for h in 0..self.num_heads {
            let start = h * d_head;
            let end = start + d_head;
            let q_h = graph::slice_cols(&q, start, end);
            let k_h = graph::slice_cols(&k, start, end);
            let v_h = graph::slice_cols(&v, start, end);
            heads.push(attention_batch(&q_h, &k_h, &v_h));
        }

        let concat = graph::concat_cols(&heads);
        let flat = graph::reshape(&concat, vec![batch * seq, d]);
        let out = graph::matmul(&flat, &self.w_o);
        graph::reshape(&out, vec![batch, seq, d])
    }
}