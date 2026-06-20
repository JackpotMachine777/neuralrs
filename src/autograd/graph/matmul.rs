use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Matrix multiplication of two 2-D nodes: `[m, k] x [k, n] -> [m, n]`.
///
/// This is the autograd-aware version (it builds a graph node); the plain,
/// faster matmul without gradient tracking lives in `ops::matmul`.
///
/// Backward uses the standard matmul gradient rules: the gradient w.r.t. `a` is
/// `grad @ b^T`, and the gradient w.r.t. `b` is `a^T @ grad` — both written out
/// here as explicit loops.
pub fn matmul(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let m = a.borrow().shape[0];
    let k = a.borrow().shape[1];
    let n = b.borrow().shape[1];

    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for t in 0..k {
                data[i * n + j] += a.borrow().data[i * k + t] * b.borrow().data[t * n + j];
            }
        }
    }

    let shape = vec![m, n];
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];
    }

    let a_clone = a.clone();
    let b_clone = b.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let m = a_clone.borrow().shape[0];
        let k = a_clone.borrow().shape[1];
        let n = b_clone.borrow().shape[1];

        for i in 0..m {
            for t in 0..k {
                let mut g = 0.0;
                
                for j in 0..n {
                    g += grad[i * n + j] * b_clone.borrow().data[t * n + j];
                }
                a_clone.borrow_mut().grad[i * k + t] += g;
            }
        }

        for t in 0..k {
            for j in 0..n {
                let mut g = 0.0;

                for i in 0..m {
                    g += a_clone.borrow().data[i * k + t] * grad[i * n + j];
                }
                b_clone.borrow_mut().grad[t * n + j] += g;
            }
        }
    }));

    n
}