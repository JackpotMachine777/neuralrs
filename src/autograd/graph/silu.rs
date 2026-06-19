use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

pub fn silu(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let input = a.borrow().data.clone();
    let data: Vec<f32> = if input.len() > PAR_THRESHOLD {
        input.par_iter().map(|&x| x * sig(x)).collect()
    } else {
        input.iter().map(|&x| x * sig(x)).collect()
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
                let s = 1.0 / (1.0 + (-x[i]).exp());
                let d = s + x[i] * s * (1.0 - s);
                grad[i] * d
            }).collect()
        } else {
            (0..grad.len()).map(|i| {
                let s = 1.0 / (1.0 + (-x[i]).exp());
                let d = s + x[i] * s * (1.0 - s);
                grad[i] * d
            }).collect()
        };
        let mut ig = a_clone.borrow_mut();
        for i in 0..local.len() { ig.grad[i] += local[i]; }
    }));

    n
}