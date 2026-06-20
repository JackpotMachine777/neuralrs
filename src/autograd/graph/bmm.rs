use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Batched matrix multiply: `[batch, m, k] x [batch, k, n] -> [batch, m, n]`.
///
/// Does an independent matmul for each item in the batch. Used by batched
/// attention, where every sequence in the batch gets its own Q/K/V products.
///
/// Backward applies the same matmul gradient rules as [`matmul`], once per batch
/// item.
///
/// [`matmul`]: crate::autograd::graph::matmul::matmul
pub fn bmm(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let a_shape = a.borrow().shape.clone();
    let b_shape = b.borrow().shape.clone();

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();

    let mut data = vec![0.0; batch * m * n];

    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let o_off = bi * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for t in 0..k {
                    sum += a_data[a_off + i * k + t] * b_data[b_off + t * n + j];
                }
                data[o_off + i * n + j] = sum;
            }
        }
    }

    let out = Node::new(data, vec![batch, m, n]);

    {
        let mut node = out.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];
    }

    let a_clone = a.clone();
    let b_clone = b.clone();

    out.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let a_data = a_clone.borrow().data.clone();
        let b_data = b_clone.borrow().data.clone();

        let mut a_grad = vec![0.0; batch * m * k];
        let mut b_grad = vec![0.0; batch * k * n];

        for bi in 0..batch {
            let a_off = bi * m * k;
            let b_off = bi * k * n;
            let o_off = bi * m * n;

            for i in 0..m {
                for t in 0..k {
                    let mut g = 0.0;
                    for j in 0..n {
                        g += grad[o_off + i * n + j] * b_data[b_off + t * n + j];
                    }
                    a_grad[a_off + i * k + t] += g;
                }
            }

            for t in 0..k {
                for j in 0..n {
                    let mut g = 0.0;
                    for i in 0..m {
                        g += a_data[a_off + i * k + t] * grad[o_off + i * n + j];
                    }
                    b_grad[b_off + t * n + j] += g;
                }
            }
        }

        {
            let mut ag = a_clone.borrow_mut();
            for idx in 0..a_grad.len() { ag.grad[idx] += a_grad[idx]; }
        }
        {
            let mut bg = b_clone.borrow_mut();
            for idx in 0..b_grad.len() { bg.grad[idx] += b_grad[idx]; }
        }
    }));

    out
}