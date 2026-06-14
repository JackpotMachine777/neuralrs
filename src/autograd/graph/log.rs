use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn log(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.iter()
        .map(|&x| x.ln())
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
            a_clone.borrow_mut().grad[i] += grad[i] / x;
        }
    }));

    n
}