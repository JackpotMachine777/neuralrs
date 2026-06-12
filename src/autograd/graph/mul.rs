use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn mul(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.iter()
        .zip(b.borrow().data.iter())
        .map(|(x, y)| x * y)
        .collect();
    
    let shape = a.borrow().shape.clone();
    
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];
    }

    let a_clone = a.clone();
    let b_clone = b.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for i in 0..grad.len() {
            a_clone.borrow_mut().grad[i] += grad[i] * b_clone.borrow().data[i];
            b_clone.borrow_mut().grad[i] += grad[i] * a_clone.borrow().data[i];
        }
    }));

    n
}