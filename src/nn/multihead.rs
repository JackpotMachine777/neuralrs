use crate::tensor::Tensor;
use crate::autograd::node::Node;
use crate::autograd::graph;
use crate::nn::attention::attention;
use std::rc::Rc;
use std::cell::RefCell;

pub struct MultiHeadAttention {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    pub d_model: usize,
    pub num_heads: usize,
    pub w_q_node: Option<Rc<RefCell<Node>>>,
    pub w_k_node: Option<Rc<RefCell<Node>>>,
    pub w_v_node: Option<Rc<RefCell<Node>>>,
    pub w_o_node: Option<Rc<RefCell<Node>>>,
}

impl MultiHeadAttention {
    pub fn forward(&mut self, x: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        if self.w_q_node.is_none() {
            self.w_q_node = Some(Node::new(self.w_q.storage.data.clone(), self.w_q.shape.clone()));
            self.w_k_node = Some(Node::new(self.w_k.storage.data.clone(), self.w_k.shape.clone()));
            self.w_v_node = Some(Node::new(self.w_v.storage.data.clone(), self.w_v.shape.clone()));
            self.w_o_node = Some(Node::new(self.w_o.storage.data.clone(), self.w_o.shape.clone()));
        }
        let w_q = self.w_q_node.clone().unwrap();
        let w_k = self.w_k_node.clone().unwrap();
        let w_v = self.w_v_node.clone().unwrap();
        let w_o = self.w_o_node.clone().unwrap();

        let q = graph::matmul(x.clone(), w_q);
        let k = graph::matmul(x.clone(), w_k);
        let v = graph::matmul(x.clone(), w_v);

        let d_head = self.d_model / self.num_heads;

        let mut heads = Vec::new();
        for h in 0..self.num_heads {
            let start = h * d_head;
            let end = start + d_head;

            let q_h = graph::slice_cols(q.clone(), start, end);
            let k_h = graph::slice_cols(k.clone(), start, end);
            let v_h = graph::slice_cols(v.clone(), start, end);

            let head_out = attention(q_h, k_h, v_h);
            heads.push(head_out);
        }

        let concat = graph::concat_cols(heads);

        graph::matmul(concat, w_o)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o]
    }

    pub fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.grad = vec![0.0; p.storage.data.len()];
        }
        self.w_q_node = None;
        self.w_k_node = None;
        self.w_v_node = None;
        self.w_o_node = None;
    }

    pub fn sync_grads(&mut self) {
        if let Some(w) = &self.w_q_node { self.w_q.grad = w.borrow().grad.clone(); }
        if let Some(w) = &self.w_k_node { self.w_k.grad = w.borrow().grad.clone(); }
        if let Some(w) = &self.w_v_node { self.w_v.grad = w.borrow().grad.clone(); }
        if let Some(w) = &self.w_o_node { self.w_o.grad = w.borrow().grad.clone(); }
    }
}