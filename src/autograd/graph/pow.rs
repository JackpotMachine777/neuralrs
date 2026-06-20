use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

/// Raises each element to a fixed power `p`. Backward multiplies the gradient by
/// `p · x^(p-1)`.
pub fn pow(a: Rc<RefCell<Node>>, p: f32) -> Rc<RefCell<Node>> {
    let input = a.borrow().data.clone();
    let data: Vec<f32> = if input.len() > PAR_THRESHOLD {
        input.par_iter().map(|&x| x.powf(p)).collect()
    } else {
        input.iter().map(|&x| x.powf(p)).collect()
    };

    let shape = a.borrow().shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let x = a_clone.borrow().data.clone();
        let local: Vec<f32> = if grad.len() > PAR_THRESHOLD {
            (0..grad.len()).into_par_iter().map(|i| grad[i] * (p * x[i].powf(p - 1.0))).collect()
        } else {
            (0..grad.len()).map(|i| grad[i] * (p * x[i].powf(p - 1.0))).collect()
        };
        let mut ig = a_clone.borrow_mut();
        for i in 0..local.len() { ig.grad[i] += local[i]; }
    }));

    n
}