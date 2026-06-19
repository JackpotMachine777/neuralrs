use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

pub fn gelu(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let c = 0.7978845608_f32;
    let input = a.borrow().data.clone();

    let out: Vec<f32> = if input.len() > PAR_THRESHOLD {
        input.par_iter().map(|&x| {
            let inner = c * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect()
    } else {
        input.iter().map(|&x| {
            let inner = c * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect()
    };

    let shape = a.borrow().shape.clone();
    let n = Node::new(out, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let x = a_clone.borrow().data.clone();
        let local: Vec<f32> = if grad.len() > PAR_THRESHOLD {
            (0..grad.len()).into_par_iter().map(|i| {
                let xi = x[i];
                let inner = c * (xi + 0.044715 * xi * xi * xi);
                let t = inner.tanh();
                let dinner = c * (1.0 + 3.0 * 0.044715 * xi * xi);
                let dgelu = 0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * dinner;
                dgelu * grad[i]
            }).collect()
        } else {
            (0..grad.len()).map(|i| {
                let xi = x[i];
                let inner = c * (xi + 0.044715 * xi * xi * xi);
                let t = inner.tanh();
                let dinner = c * (1.0 + 3.0 * 0.044715 * xi * xi);
                let dgelu = 0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * dinner;
                dgelu * grad[i]
            }).collect()
        };
        let mut ig = a_clone.borrow_mut();
        for i in 0..local.len() { ig.grad[i] += local[i]; }
    }));

    n
}