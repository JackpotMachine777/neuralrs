use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

pub fn mse_node(pred: Rc<RefCell<Node>>, target: Rc<RefCell<Node>>) -> f32 {
    let p = pred.borrow();
    let t = target.borrow();
    let mut res = 0.0;
    
    for i in 0..p.data.len() {
        let diff = p.data[i] - t.data[i];
        res += diff * diff;
    }

    res / p.data.len() as f32
}

pub fn mse_grad_node(pred: Rc<RefCell<Node>>, target: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let p = pred.borrow();
    let t = target.borrow();
    let n = p.data.len();

    let data: Vec<f32> = (0..n).map(|i| 2.0 * (p.data[i] - t.data[i]) / n as f32).collect();
    let shape = p.shape.clone();

    Node::new(data, shape)
}