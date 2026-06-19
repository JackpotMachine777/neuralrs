use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn concat_cols(parts: Vec<Rc<RefCell<Node>>>) -> Rc<RefCell<Node>> {
    let widths: Vec<usize> = parts.iter().map(|p| *p.borrow().shape.last().unwrap()).collect();
    let total_cols: usize = widths.iter().sum();

    let first_len = parts[0].borrow().data.len();
    let rows = first_len / widths[0];

    let mut out = vec![0.0; rows * total_cols];
    let mut col_offset = 0;
    for (idx, part) in parts.iter().enumerate() {
        let pw = widths[idx];
        let pdata = part.borrow().data.clone();
        for r in 0..rows {
            for c in 0..pw {
                out[r * total_cols + (col_offset + c)] = pdata[r * pw + c];
            }
        }
        col_offset += pw;
    }

    let mut out_shape = parts[0].borrow().shape.clone();
    let last = out_shape.len() - 1;
    out_shape[last] = total_cols;

    let result = Node::new(out, out_shape);
    {
        let mut node = result.borrow_mut();
        node.parents = parts.clone();
    }

    let parts_clone = parts.clone();
    let widths_clone = widths.clone();
    result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let mut col_offset = 0;
        for (idx, part) in parts_clone.iter().enumerate() {
            let pw = widths_clone[idx];
            for r in 0..rows {
                for c in 0..pw {
                    part.borrow_mut().grad[r * pw + c] += grad[r * total_cols + (col_offset + c)];
                }
            }
            col_offset += pw;
        }
        let _ = col_offset;
    }));

    result
}