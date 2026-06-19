use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

pub fn elu(a: Rc<RefCell<Node>>, alpha: f32) -> Rc<RefCell<Node>> {
    let input = a.borrow().data.clone();
    let data: Vec<f32> = if input.len() > PAR_THRESHOLD {
        input.par_iter().map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }).collect()
    } else {
        input.iter().map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }).collect()
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
            (0..grad.len()).into_par_iter().map(|i| {
                let d = if x[i] > 0.0 { 1.0 } else { alpha * x[i].exp() };
                grad[i] * d
            }).collect()
        } else {
            (0..grad.len()).map(|i| {
                let d = if x[i] > 0.0 { 1.0 } else { alpha * x[i].exp() };
                grad[i] * d
            }).collect()
        };
        let mut ig = a_clone.borrow_mut();
        for i in 0..local.len() { ig.grad[i] += local[i]; }
    }));

    n
}