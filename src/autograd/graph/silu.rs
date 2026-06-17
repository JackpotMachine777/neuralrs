use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn silu(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let data: Vec<f32> = a.borrow().data.iter()
        .map(|&x| x * sig(x))
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
            let x = a_clone.borrow().data[i];
            let s = 1.0 / (1.0 + (-x).exp());
            let d = s + x * s * (1.0 - s);
            a_clone.borrow_mut().grad[i] += grad[i] * d;
        }
    }));

    n
} 