//! LSTM cell on the GPU, composed from resident graph ops (matmul, add, mul,
//! sigmoid, tanh). Weights are resident nodes; autograd flows through the
//! composition. Mirrors the CPU `LSTMCell`.

use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::node::Node;
use crate::cuda::graph;

type N = Rc<RefCell<Node>>;

/// An LSTM cell over resident weight nodes: four gates (forget `f`, input `i`,
/// output `o`, candidate `g`), each with input weights `w_*`, recurrent weights
/// `u_*`, and bias `b_*`.
pub struct LSTMCell {
    pub w_f: N, pub u_f: N, pub b_f: N,
    pub w_i: N, pub u_i: N, pub b_i: N,
    pub w_o: N, pub u_o: N, pub b_o: N,
    pub w_g: N, pub u_g: N, pub b_g: N,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl LSTMCell {
    /// One timestep: `x`, previous hidden `h_prev`, previous cell `c_prev`
    /// (states `[batch, hidden]`) -> `(h_new, c_new)`.
    pub fn step(&self, x: &N, h_prev: &N, c_prev: &N) -> (N, N) {
        let gate = |x: &N, h: &N, w: &N, u: &N, b: &N| -> N {
            let xw = graph::matmul(x, w);
            let hu = graph::matmul(h, u);
            let s = graph::add(&xw, &hu);
            graph::add(&s, b)
        };

        let f = graph::sigmoid(&gate(x, h_prev, &self.w_f, &self.u_f, &self.b_f));
        let i = graph::sigmoid(&gate(x, h_prev, &self.w_i, &self.u_i, &self.b_i));
        let o = graph::sigmoid(&gate(x, h_prev, &self.w_o, &self.u_o, &self.b_o));
        let g = graph::tanh(&gate(x, h_prev, &self.w_g, &self.u_g, &self.b_g));

        let fc = graph::mul(&f, c_prev);
        let ig = graph::mul(&i, &g);
        let c_new = graph::add(&fc, &ig);

        let c_tanh = graph::tanh(&c_new);
        let h_new = graph::mul(&o, &c_tanh);

        (h_new, c_new)
    }
}