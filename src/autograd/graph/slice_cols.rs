use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn slice_cols(a: Rc<RefCell<Node>>, col_start: usize, col_end: usize) -> Rc<RefCell<Node>> {
    let data = a.borrow().data.clone();
    let shape = a.borrow().shape.clone();
    let rows = shape[0];
    let total_cols = shape[1];
    let slice_w = col_end - col_start;

    let mut out = vec![0.0; rows * slice_w];
    for r in 0..rows {
        for c in 0..slice_w {
            out[r * slice_w + c] = data[r * total_cols + (col_start + c)];
        }
    }

    let result = Node::new(out, vec![rows, slice_w]);
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