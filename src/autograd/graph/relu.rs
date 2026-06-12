use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn relu(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
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
            let g = if a_clone.borrow().data[i] > 0.0 { grad[i] } else { 0.0 };
            a_clone.borrow_mut().grad[i] += g;
        }
    }));

    n
}