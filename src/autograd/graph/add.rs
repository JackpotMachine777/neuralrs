use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn add(a: Rc<RefCell<Node>>, b: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let a_data = a.borrow().data.clone(); 
    let a_shape = a.borrow().shape.clone();
    let b_data = b.borrow().data.clone();
    let b_shape = b.borrow().shape.clone();

    let broadcast = a_shape.len() == 2 && b_shape.len() == 1 && a_shape[1] == b_shape[0];

    let data: Vec<f32> = if broadcast {
        let batch = a_shape[0];
        let features = a_shape[1];
        let mut out = vec![0.0; a_data.len()];
        for bi in 0..batch {
            for f in 0..features {
                let idx = bi * features + f;
                out[idx] = a_data[idx] + b_data[f];
            }
        }
        out
    } else {
        a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect()
    };

    let shape = a_shape.clone();
    let n = Node::new(data, shape);

    {
        let mut node = n.borrow_mut();
        node.parents = vec![a.clone(), b.clone()];
    }

    let a_clone = a.clone();
    let b_clone = b.clone();
    let a_shape_c = a_shape.clone();
    let b_shape_c = b_shape.clone();

    n.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let broadcast = a_shape_c.len() == 2 && b_shape_c.len() == 1 && a_shape_c[1] == b_shape_c[0];

        if broadcast {
            let batch = a_shape_c[0];
            let features = a_shape_c[1];

            for i in 0..grad.len() {
                a_clone.borrow_mut().grad[i] += grad[i];
            }

            for bi in 0..batch {
                for f in 0..features {
                    b_clone.borrow_mut().grad[f] += grad[bi * features + f];
                }
            }
        } else {
            for i in 0..grad.len() {
                a_clone.borrow_mut().grad[i] += grad[i];
                b_clone.borrow_mut().grad[i] += grad[i];
            }
        }
    }));

    n
}