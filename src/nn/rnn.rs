use crate::tensor::Tensor;
use crate::autograd::node::Node;
use crate::autograd::graph;
use std::rc::Rc;
use std::cell::RefCell;

pub struct RNNCell {
    pub w_xh: Tensor,
    pub w_hh: Tensor,
    pub bias: Tensor,
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_xh_node: Option<Rc<RefCell<Node>>>,
    pub w_hh_node: Option<Rc<RefCell<Node>>>,
    pub bias_node: Option<Rc<RefCell<Node>>>,
}

impl RNNCell {
    pub fn step(
        &mut self,
        x: Rc<RefCell<Node>>,
        h_prev: Rc<RefCell<Node>>,
    ) -> Rc<RefCell<Node>> {
        if self.w_xh_node.is_none() {
            self.w_xh_node = Some(Node::new(self.w_xh.storage.data.clone(), self.w_xh.shape.clone()));
            self.w_hh_node = Some(Node::new(self.w_hh.storage.data.clone(), self.w_hh.shape.clone()));
            self.bias_node = Some(Node::new(self.bias.storage.data.clone(), self.bias.shape.clone()));
        }

        let w_xh = self.w_xh_node.clone().unwrap();
        let w_hh = self.w_hh_node.clone().unwrap();
        let b = self.bias_node.clone().unwrap();

        let xh = graph::matmul(x, w_xh);
        let hh = graph::matmul(h_prev, w_hh);
        let sum = graph::add(xh, hh);
        let with_bias = graph::add(sum, b);

        graph::tanh(with_bias)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w_xh, &mut self.w_hh, &mut self.bias]
    }

    pub fn zero_grad(&mut self) {
        self.w_xh.grad = vec![0.0; self.w_xh.storage.data.len()];
        self.w_hh.grad = vec![0.0; self.w_hh.storage.data.len()];
        self.bias.grad = vec![0.0; self.bias.storage.data.len()];

        self.w_xh_node = None;
        self.w_hh_node = None;
        self.bias_node = None;
    }

    pub fn sync_grads(&mut self) {
        if let Some(w) = &self.w_xh_node {
            self.w_xh.grad = w.borrow().grad.clone();
        }
        if let Some(w) = &self.w_hh_node {
            self.w_hh.grad = w.borrow().grad.clone();
        }
        if let Some(b) = &self.bias_node {
            self.bias.grad = b.borrow().grad.clone();
        }
    }
}