use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

/// ReLU activation: keeps positive values, clamps everything else to zero.
///
/// Backward passes the gradient through wherever the input was positive, and
/// blocks it (zero) where the input was zero or negative — a "dead" unit gets no
/// gradient.
pub fn relu(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let input = a.borrow().data.clone();

    let data: Vec<f32> = if input.len() > PAR_THRESHOLD {
        input.par_iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
    } else {
        input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
    };

    let shape = a.borrow().shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let a_data = a_clone.borrow().data.clone();
        let local: Vec<f32> = if grad.len() > PAR_THRESHOLD {
            (0..grad.len()).into_par_iter()
                .map(|i| if a_data[i] > 0.0 { grad[i] } else { 0.0 })
                .collect()
        } else {
            (0..grad.len())
                .map(|i| if a_data[i] > 0.0 { grad[i] } else { 0.0 })
                .collect()
        };
        let mut ig = a_clone.borrow_mut();
        for i in 0..local.len() {
            ig.grad[i] += local[i];
        }
    }));

    n
}