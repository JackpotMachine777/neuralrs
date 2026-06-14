use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn softmax(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.clone();

    let exps: Vec<f32> = data.iter().map(|&x| x.exp()).collect();
    let sum: f32 = exps.iter().sum();
    let out: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let shape = a.borrow().shape.clone();
    let n = Node::new(out, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    let out_clone = n.borrow().data.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let dot: f32 = out_clone.iter().zip(grad.iter()).map(|(o, g)| o * g).sum();

        for i in 0..grad.len() {
            let g = out_clone[i] * (grad[i] - dot);
            a_clone.borrow_mut().grad[i] += g;
        }
    }));

    n
}