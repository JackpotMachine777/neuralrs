use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Softmax over the last dimension, turning each row into a probability
/// distribution (positive values that sum to 1).
///
/// Works on any shape — it treats the data as rows of length `last_dim` and
/// softmaxes each one. Subtracts the row max before exponentiating, which avoids
/// overflow without changing the result.
///
/// Backward uses the softmax Jacobian trick: for each row, `grad_in = out * (grad
/// - sum(out * grad))`, which is the clean closed form instead of building the
/// full Jacobian matrix.
pub fn softmax(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data: Vec<f32> = a.borrow().data.clone();
    let shape = a.borrow().shape.clone();

    let width = *shape.last().unwrap();
    let rows = data.len() / width;

    let mut out = vec![0.0; data.len()];

    for r in 0..rows {
        let start = r * width;
        let row = &data[start..start + width];

        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        for f in 0..width {
            out[start + f] = exps[f] / sum;
        }
    }

    let n = Node::new(out, shape.clone());

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    let out_clone = n.borrow().data.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for r in 0..rows {
            let start = r * width;
            let mut dot = 0.0;

            for f in 0..width {
                dot += out_clone[start + f] * grad[start + f];
            }

            for f in 0..width {
                let idx = start + f;
                let g = out_clone[idx] * (grad[idx] - dot);
                a_clone.borrow_mut().grad[idx] += g;
            }
        }
    }));

    n
}