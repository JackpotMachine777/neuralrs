//! Scaled dot-product attention on the GPU, composed from resident graph ops
//! (transpose, matmul/bmm, scale, softmax). No new kernels — autograd flows
//! through the composed primitives. Mirrors the CPU `attention`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;

/// Single-sequence attention `softmax(Q·Kᵀ / √d) · V` over resident `[seq, d]` nodes.
pub fn attention(q: &Rc<RefCell<Node>>, k: &Rc<RefCell<Node>>, v: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let d = q.borrow().shape[1];
    let scale_factor = 1.0 / (d as f32).sqrt();

    let k_t = graph::transpose(k);
    let scores = graph::matmul(q, &k_t);
    let scaled = graph::scale(&scores, scale_factor);
    let weights = graph::softmax(&scaled);
    graph::matmul(&weights, v)
}

/// Batched attention over resident `[batch, seq, d]` nodes, using batched matmul.
pub fn attention_batch(q: &Rc<RefCell<Node>>, k: &Rc<RefCell<Node>>, v: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let d = q.borrow().shape[2];
    let scale_factor = 1.0 / (d as f32).sqrt();

    let k_t = graph::transpose(k);
    let scores = graph::bmm(q, &k_t);
    let scaled = graph::scale(&scores, scale_factor);
    let weights = graph::softmax(&scaled);
    graph::bmm(&weights, v)
}