//! Sinusoidal positional encoding on the GPU. Precomputes the encoding on the
//! host up to `max_len`; `forward` uploads the right slice and adds it to the
//! resident input. Mirrors the CPU `PositionalEncoding`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;
use crate::cuda::runtime::to_cuda;

/// Fixed sinusoidal positional encoding, precomputed up to `max_len`.
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
                pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        Self { d_model, max_len, pe }
    }

    /// Adds positional encoding to a resident `[seq, d_model]` input.
    ///
    /// # Panics
    /// If `seq > max_len`, or the input is not on the GPU.
    pub fn forward(&self, x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let seq_len = x.borrow().shape[0];
        assert!(seq_len <= self.max_len, "sequence longer than max_len");

        let pe_slice: Vec<f32> = self.pe[0..seq_len * self.d_model].to_vec();
        let pe_node = Node::new(pe_slice, vec![seq_len, self.d_model]);
        to_cuda(&pe_node);
        graph::add(x, &pe_node)
    }
}