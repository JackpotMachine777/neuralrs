use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

/// Multiplies every element by a fixed scalar `s`. Backward simply scales the
/// gradient by the same `s`.
pub fn scale(a: Rc<RefCell<Node>>, s: f32) -> Rc<RefCell<Node>> {
    let input = a.borrow().data.clone();
    let data: Vec<f32> = if input.len() > PAR_THRESHOLD {
        input.par_iter().map(|&x| x * s).collect()
    } else {
        input.iter().map(|&x| x * s).collect()
    };

    let shape = a.borrow().shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let local: Vec<f32> = if grad.len() > PAR_THRESHOLD {
            (0..grad.len()).into_par_iter().map(|i| grad[i] * s).collect()
        } else {
            (0..grad.len()).map(|i| grad[i] * s).collect()
        };
        let mut ig = a_clone.borrow_mut();
        for i in 0..local.len() { ig.grad[i] += local[i]; }
    }));

    n
}