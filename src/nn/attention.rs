use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;
use crate::autograd::graph;
use crate::autograd::graph::bmm::bmm;

/// Scaled dot-product attention for a single sequence: `softmax(Q·Kᵀ / √d) · V`.
pub fn attention(
    q: Rc<RefCell<Node>>,
    k: Rc<RefCell<Node>>,
    v: Rc<RefCell<Node>>,
) -> Rc<RefCell<Node>> {
    let d = q.borrow().shape[1];
    let scale_factor = 1.0 / (d as f32).sqrt();

    let k_t = graph::transpose(k);
    let scores = graph::matmul(q, k_t);
    let scaled = graph::scale(scores, scale_factor);
    let weights = graph::softmax(scaled);

    graph::matmul(weights, v)
}

/// Batched scaled dot-product attention over `[batch, seq, d]`, using batched
/// matmul so every sequence in the batch is handled at once.
pub fn attention_batch(
    q: Rc<RefCell<Node>>,
    k: Rc<RefCell<Node>>,
    v: Rc<RefCell<Node>>,
) -> Rc<RefCell<Node>> {
    let d = q.borrow().shape[2];
    let scale_factor = 1.0 / (d as f32).sqrt();

    let k_t = graph::transpose(k);
    let scores = bmm(q, k_t);
    let scaled = graph::scale(scores, scale_factor);
    let weights = graph::softmax(scaled);
    bmm(weights, v)
}