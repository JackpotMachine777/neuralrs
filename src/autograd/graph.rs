use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn add(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.iter()
        .zip(b.borrow().data.iter())
        .map(|(x, y)| x + y)
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
            a_clone.borrow_mut().grad[i] += grad[i];
            b_clone.borrow_mut().grad[i] += grad[i];
        }
    }));

    n
}

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