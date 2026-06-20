use crate::autograd::node::Node;
use crate::autograd::graph;
use std::rc::Rc;
use std::cell::RefCell;

/// Sinusoidal positional encoding — adds position information to token
/// embeddings.
///
/// Self-attention has no built-in sense of order, so this adds a fixed pattern
/// of sines and cosines (different frequency per dimension) that tells the model
/// where each token sits in the sequence. Precomputed once up to `max_len`;
/// `forward` just adds the right slice to the input.
pub struct PositionalEncoding {
    pub d_model: usize,
    pub max_len: usize,
    pe: Vec<f32>,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut pe = vec![0.0; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let exponent = (2 * (i / 2)) as f32 / d_model as f32;
                let freq = 1.0 / 10000_f32.powf(exponent);
                let angle = pos as f32 * freq;

                pe[pos * d_model + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos() 
                };
            }
        }

        PositionalEncoding { d_model, max_len, pe }
    }

    pub fn forward(&self, x: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let seq_len = x.borrow().shape[0];
        assert!(seq_len <= self.max_len, "sequence longer than max_len");

        let pe_slice: Vec<f32> = self.pe[0..seq_len * self.d_model].to_vec();
        let pe_node = Node::new(pe_slice, vec![seq_len, self.d_model]);

        graph::add(x, pe_node)
    }
}