use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Transposes the last two dimensions of a node, batched over any leading dims.
///
/// So `[rows, cols]` becomes `[cols, rows]`, and `[batch, rows, cols]` becomes
/// `[batch, cols, rows]`. Used in attention to line up keys for the Q·Kᵀ product.
///
/// Backward just transposes the incoming gradient back the same way.
pub fn transpose(a: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let data = a.borrow().data.clone();
    let shape = a.borrow().shape.clone();

    let ndim = shape.len();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let batch = data.len() / (rows * cols);

    let mut out = vec![0.0; data.len()];

    for b in 0..batch {
        let off = b * rows * cols;
        for i in 0..rows {
            for j in 0..cols {
                out[off + j * rows + i] = data[off + i * cols + j];
            }
        }
    }

    let mut out_shape = shape.clone();
    out_shape[ndim - 2] = cols;
    out_shape[ndim - 1] = rows;

    let result = Node::new(out, out_shape);

    {
        let mut node = result.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for b in 0..batch {
            let off = b * rows * cols;
            for i in 0..rows {
                for j in 0..cols {
                    a_clone.borrow_mut().grad[off + i * cols + j] += grad[off + j * rows + i];
                }
            }
        }
    }));

    result
}