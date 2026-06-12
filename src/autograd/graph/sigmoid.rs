use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn sigmoid(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();

    let shape = a.borrow().shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for i in 0..grad.len() {
            let s = 1.0 / (1.0 + (-a_clone.borrow().data[i]).exp());
            let g = s * (1.0 - s) * grad[i];
            a_clone.borrow_mut().grad[i] += g;
        }
    }));

    n
}