use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn gelu(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let c = 0.7978845608_f32;
    let data = a.borrow().data.clone();

    let out: Vec<f32> = data.iter().map(|&x| {
        let inner = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }).collect();

    let shape = a.borrow().shape.clone();
    let n = Node::new(out, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for i in 0..grad.len() {
            let x = a_clone.borrow().data[i];
            let inner = c * (x + 0.044715 * x * x * x);
            let t = inner.tanh();
            let dinner = c * (1.0 + 3.0 * 0.044715 * x * x);
            let dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner;
            a_clone.borrow_mut().grad[i] += dgelu * grad[i];
        }
    }));

    n
}