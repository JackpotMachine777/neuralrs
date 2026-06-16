use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn transpose(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data = a.borrow().data.clone();
    let shape = a.borrow().shape.clone();
    let rows = shape[0];
    let cols = shape[1];

    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }

    let result = Node::new(out, vec![cols, rows]);

    {
        let mut node = result.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for i in 0..rows {
            for j in 0..cols {
                a_clone.borrow_mut().grad[i * cols + j] += grad[j * rows + i];
            }
        }
    }));

    result
}