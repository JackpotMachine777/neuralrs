use crate::tensor::Tensor;
use crate::autograd::node::Node;
use crate::autograd::graph;
use std::rc::Rc;
use std::cell::RefCell;

pub struct LSTMCell {
    pub w_f: Tensor,
    pub u_f: Tensor,
    pub b_f: Tensor,
    pub w_i: Tensor,
    pub u_i: Tensor,
    pub b_i: Tensor,
    pub w_o: Tensor,
    pub u_o: Tensor,
    pub b_o: Tensor,
    pub w_g: Tensor,
    pub u_g: Tensor,
    pub b_g: Tensor,
    pub input_size: usize,
    pub hidden_size: usize,
    pub nodes: Option<LSTMNodes>,
}

pub struct LSTMNodes {
    pub w_f: Rc<RefCell<Node>>, pub u_f: Rc<RefCell<Node>>, pub b_f: Rc<RefCell<Node>>,
    pub w_i: Rc<RefCell<Node>>, pub u_i: Rc<RefCell<Node>>, pub b_i: Rc<RefCell<Node>>,
    pub w_o: Rc<RefCell<Node>>, pub u_o: Rc<RefCell<Node>>, pub b_o: Rc<RefCell<Node>>,
    pub w_g: Rc<RefCell<Node>>, pub u_g: Rc<RefCell<Node>>, pub b_g: Rc<RefCell<Node>>,
}

impl LSTMCell {
    pub fn step(
        &mut self,
        x: Rc<RefCell<Node>>,
        h_prev: Rc<RefCell<Node>>,
        c_prev: Rc<RefCell<Node>>,
    ) -> (Rc<RefCell<Node>>, Rc<RefCell<Node>>) {
        if self.nodes.is_none() {
            self.nodes = Some(LSTMNodes {
                w_f: Node::new(self.w_f.storage.data.clone(), self.w_f.shape.clone()),
                u_f: Node::new(self.u_f.storage.data.clone(), self.u_f.shape.clone()),
                b_f: Node::new(self.b_f.storage.data.clone(), self.b_f.shape.clone()),
                w_i: Node::new(self.w_i.storage.data.clone(), self.w_i.shape.clone()),
                u_i: Node::new(self.u_i.storage.data.clone(), self.u_i.shape.clone()),
                b_i: Node::new(self.b_i.storage.data.clone(), self.b_i.shape.clone()),
                w_o: Node::new(self.w_o.storage.data.clone(), self.w_o.shape.clone()),
                u_o: Node::new(self.u_o.storage.data.clone(), self.u_o.shape.clone()),
                b_o: Node::new(self.b_o.storage.data.clone(), self.b_o.shape.clone()),
                w_g: Node::new(self.w_g.storage.data.clone(), self.w_g.shape.clone()),
                u_g: Node::new(self.u_g.storage.data.clone(), self.u_g.shape.clone()),
                b_g: Node::new(self.b_g.storage.data.clone(), self.b_g.shape.clone()),
            });
        }

        let n = self.nodes.as_ref().unwrap();

        let gate = |x: &Rc<RefCell<Node>>, h: &Rc<RefCell<Node>>, 
                    w: &Rc<RefCell<Node>>, u: &Rc<RefCell<Node>>, b: &Rc<RefCell<Node>>| {
            let xw = graph::matmul(x.clone(), w.clone());
            let hu = graph::matmul(h.clone(), u.clone());
            let s = graph::add(xw, hu);
            graph::add(s, b.clone())
        };

        let f = graph::sigmoid(gate(&x, &h_prev, &n.w_f, &n.u_f, &n.b_f));
        let i = graph::sigmoid(gate(&x, &h_prev, &n.w_i, &n.u_i, &n.b_i));
        let o = graph::sigmoid(gate(&x, &h_prev, &n.w_o, &n.u_o, &n.b_o));
        let g = graph::tanh(gate(&x, &h_prev, &n.w_g, &n.u_g, &n.b_g));

        let fc = graph::mul(f, c_prev);
        let ig = graph::mul(i, g);
        let c_new = graph::add(fc, ig);

        let c_tanh = graph::tanh(c_new.clone());
        let h_new = graph::mul(o, c_tanh);

        (h_new, c_new)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_f, &mut self.u_f, &mut self.b_f,
            &mut self.w_i, &mut self.u_i, &mut self.b_i,
            &mut self.w_o, &mut self.u_o, &mut self.b_o,
            &mut self.w_g, &mut self.u_g, &mut self.b_g,
        ]
    }

    pub fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.grad = vec![0.0; p.storage.data.len()];
        }
        self.nodes = None;
    }

    pub fn sync_grads(&mut self) {
        if let Some(n) = &self.nodes {
            self.w_f.grad = n.w_f.borrow().grad.clone();
            self.u_f.grad = n.u_f.borrow().grad.clone();
            self.b_f.grad = n.b_f.borrow().grad.clone();
            self.w_i.grad = n.w_i.borrow().grad.clone();
            self.u_i.grad = n.u_i.borrow().grad.clone();
            self.b_i.grad = n.b_i.borrow().grad.clone();
            self.w_o.grad = n.w_o.borrow().grad.clone();
            self.u_o.grad = n.u_o.borrow().grad.clone();
            self.b_o.grad = n.b_o.borrow().grad.clone();
            self.w_g.grad = n.w_g.borrow().grad.clone();
            self.u_g.grad = n.u_g.borrow().grad.clone();
            self.b_g.grad = n.b_g.borrow().grad.clone();
        }
    }
}