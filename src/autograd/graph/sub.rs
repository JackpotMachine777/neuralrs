use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 8192;

/// Subtracts two nodes element-wise. Backward sends `+grad` to the first operand
/// and `-grad` to the second.
pub fn sub(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();

    let data: Vec<f32> = if a_data.len() > PAR_THRESHOLD {
        a_data.par_iter().zip(b_data.par_iter()).map(|(x, y)| x - y).collect()
    } else {
        a_data.iter().zip(b_data.iter()).map(|(x, y)| x - y).collect()
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
        {
            let mut ig = a_clone.borrow_mut();
            for i in 0..grad.len() { ig.grad[i] += grad[i]; }
        }
        {
            let mut ig = b_clone.borrow_mut();
            for i in 0..grad.len() { ig.grad[i] -= grad[i]; }
        }
    }));

    n
}