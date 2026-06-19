use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::nn::multihead::MultiHeadAttention;
use crate::nn::normalization::LayerNorm;
use crate::nn::linear::Linear;
use crate::autograd::node::Node;
use crate::autograd::graph;
use crate::autograd::graph::reshape::reshape;
use std::rc::Rc;
use std::cell::RefCell;

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
    pub fn forward(&mut self, x: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let shape = x.borrow().shape.clone();
        let batch = shape[0];
        let seq = shape[1];
        let d = shape[2];

        let attn_out = self.mha.forward_batch(x.clone());
        let x1 = graph::add(x, attn_out);
        let x1 = self.norm1.forward(x1);

        let flat = reshape(x1.clone(), vec![batch * seq, d]);
        let h = self.ff1.forward(flat);
        let h = graph::relu(h);
        let ff_out_2d = self.ff2.forward(h);
        let ff_out = reshape(ff_out_2d, vec![batch, seq, d]);

        let x2 = graph::add(x1, ff_out);
        self.norm2.forward(x2)
    }

    pub fn forward_no_final_norm(&mut self, x: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let shape = x.borrow().shape.clone();
        let batch = shape[0];
        let seq = shape[1];
        let d = shape[2];

        let attn_out = self.mha.forward_batch(x.clone());
        let x1 = graph::add(x, attn_out);
        let x1 = self.norm1.forward(x1);

        let flat = reshape(x1.clone(), vec![batch * seq, d]);
        let h = self.ff1.forward(flat);
        let h = graph::relu(h);
        let ff_out_2d = self.ff2.forward(h);
        let ff_out = reshape(ff_out_2d, vec![batch, seq, d]);

        graph::add(x1, ff_out)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.mha.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }

    pub fn zero_grad(&mut self) {
        self.mha.zero_grad();
        self.norm1.zero_grad();
        self.norm2.zero_grad();
        self.ff1.zero_grad();
        self.ff2.zero_grad();
    }

    pub fn sync_grads(&mut self) {
        self.mha.sync_grads();
        self.norm1.sync_grads();
        self.norm2.sync_grads();
        self.ff1.sync_grads();
        self.ff2.sync_grads();
    }
}