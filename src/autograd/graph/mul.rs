use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

pub fn mul(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();

    let data: Vec<f32> = if a_data.len() > PAR_THRESHOLD {
        a_data.par_iter().zip(b_data.par_iter()).map(|(x, y)| x * y).collect()
    } else {
        a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).collect()
    };

    let shape = a.borrow().shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];
    }

    let a_clone = a.clone();
    let b_clone = b.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let a_data = a_clone.borrow().data.clone();
        let b_data = b_clone.borrow().data.clone();

        let (ga, gb): (Vec<f32>, Vec<f32>) = if grad.len() > PAR_THRESHOLD {
            (0..grad.len()).into_par_iter()
                .map(|i| (grad[i] * b_data[i], grad[i] * a_data[i]))
                .unzip()
        } else {
            (0..grad.len())
                .map(|i| (grad[i] * b_data[i], grad[i] * a_data[i]))
                .unzip()
        };

        {
            let mut ig = a_clone.borrow_mut();
            for i in 0..ga.len() { ig.grad[i] += ga[i]; }
        }
        {
            let mut ig = b_clone.borrow_mut();
            for i in 0..gb.len() { ig.grad[i] += gb[i]; }
        }
    }));

    n
}