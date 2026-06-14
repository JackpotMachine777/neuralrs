use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn sqrt(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.iter()
        .map(|&x| x.sqrt())
        .collect();

    let shape = a.borrow().shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    let out_clone = n.borrow().data.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for i in 0..grad.len() {
            a_clone.borrow_mut().grad[i] += grad[i] / (2.0 * out_clone[i]);
        }
    }));

    n
}