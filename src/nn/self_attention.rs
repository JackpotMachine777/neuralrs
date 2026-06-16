use crate::tensor::Tensor;
use crate::autograd::node::Node;
use crate::autograd::graph;
use crate::nn::attention::attention;
use std::rc::Rc;
use std::cell::RefCell;

pub struct SelfAttention {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q_node: Option<Rc<RefCell<Node>>>,
    pub w_k_node: Option<Rc<RefCell<Node>>>,
    pub w_v_node: Option<Rc<RefCell<Node>>>,
}

impl SelfAttention {
    pub fn forward(&mut self, x: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        if self.w_q_node.is_none() {
            self.w_q_node = Some(Node::new(self.w_q.storage.data.clone(), self.w_q.shape.clone()));
            self.w_k_node = Some(Node::new(self.w_k.storage.data.clone(), self.w_k.shape.clone()));
            self.w_v_node = Some(Node::new(self.w_v.storage.data.clone(), self.w_v.shape.clone()));
        }

        let w_q = self.w_q_node.clone().unwrap();
        let w_k = self.w_k_node.clone().unwrap();
        let w_v = self.w_v_node.clone().unwrap();

        let q = graph::matmul(x.clone(), w_q);
        let k = graph::matmul(x.clone(), w_k);
        let v = graph::matmul(x.clone(), w_v);

        attention(q, k, v)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w_q, &mut self.w_k, &mut self.w_v]
    }

    pub fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.grad = vec![0.0; p.storage.data.len()];
        }

        self.w_q_node = None;
        self.w_k_node = None;
        self.w_v_node = None;
    }

    pub fn sync_grads(&mut self) {
        if let Some(w) = &self.w_q_node { self.w_q.grad = w.borrow().grad.clone(); }
        if let Some(w) = &self.w_k_node { self.w_k.grad = w.borrow().grad.clone(); }
        if let Some(w) = &self.w_v_node { self.w_v.grad = w.borrow().grad.clone(); }
    }
}