//! Transformer encoder block on the GPU: multi-head attention + position-wise
//! feed-forward, each with a residual and layer norm. Composed from resident
//! MultiHeadAttention, LayerNorm, Linear, and graph ops. Mirrors the CPU
//! `TransformerBlock`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;
use crate::cuda::nn::linear::Linear;
use crate::cuda::nn::multihead::MultiHeadAttention;
use crate::cuda::nn::normalization::LayerNorm;

/// A Transformer encoder block: `x -> MHA -> add(x) -> norm1 -> FFN -> add ->
/// norm2`, over batched input `[batch, seq, d_model]`.
pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub d_model: usize,
    pub d_ff: usize,
}

impl TransformerBlock {
    /// Full block including the final layer norm.
    pub fn forward(&self, x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let (batch, seq, d) = { let s = x.borrow(); (s.shape[0], s.shape[1], s.shape[2]) };

        let attn_out = self.mha.forward_batch(x);
        let x1 = graph::add(x, &attn_out);
        let x1 = self.norm1.forward(&x1);

        let flat = graph::reshape(&x1, vec![batch * seq, d]);
        let h = self.ff1.forward(&flat);
        let h = graph::relu(&h);
        let ff_out_2d = self.ff2.forward(&h);
        let ff_out = graph::reshape(&ff_out_2d, vec![batch, seq, d]);

        let x2 = graph::add(&x1, &ff_out);
        self.norm2.forward(&x2)
    }

    /// Same as [`forward`](Self::forward) but without the final layer norm.
    pub fn forward_no_final_norm(&self, x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let (batch, seq, d) = { let s = x.borrow(); (s.shape[0], s.shape[1], s.shape[2]) };

        let attn_out = self.mha.forward_batch(x);
        let x1 = graph::add(x, &attn_out);
        let x1 = self.norm1.forward(&x1);

        let flat = graph::reshape(&x1, vec![batch * seq, d]);
        let h = self.ff1.forward(&flat);
        let h = graph::relu(&h);
        let ff_out_2d = self.ff2.forward(&h);
        let ff_out = graph::reshape(&ff_out_2d, vec![batch, seq, d]);

        graph::add(&x1, &ff_out)
    }
}