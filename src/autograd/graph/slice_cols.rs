use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Takes a contiguous range of columns `[col_start, col_end)` from the last
/// dimension, keeping all rows.
///
/// Used to split a wide tensor into pieces (for example, slicing out each head's
/// share of the features in multi-head attention).
///
/// Backward scatters the incoming gradient back into the original column
/// positions; everything outside the slice gets no gradient.
pub fn slice_cols(a: Rc<RefCell<Node>>, col_start: usize, col_end: usize) -> Rc<RefCell<Node>> {
    let data = a.borrow().data.clone();
    let shape = a.borrow().shape.clone();

    let total_cols = *shape.last().unwrap();
    let rows = data.len() / total_cols;
    let slice_w = col_end - col_start;

    let mut out = vec![0.0; rows * slice_w];
    for r in 0..rows {
        for c in 0..slice_w {
            out[r * slice_w + c] = data[r * total_cols + (col_start + c)];
        }
    }

    let mut out_shape = shape.clone();
    let last = out_shape.len() - 1;
    out_shape[last] = slice_w;

    let result = Node::new(out, out_shape);
    {
        let mut node = result.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        for r in 0..rows {
            for c in 0..slice_w {
                a_clone.borrow_mut().grad[r * total_cols + (col_start + c)] += grad[r * slice_w + c];
            }
        }
    }));

    result
}