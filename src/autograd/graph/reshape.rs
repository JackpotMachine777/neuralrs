use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Reinterprets a node under a new shape, keeping the same underlying values.
///
/// The total number of elements has to match (it asserts this). Since reshape
/// only relabels the layout and doesn't move or change any numbers, backward
/// just passes the gradient straight through 1:1.
pub fn reshape(a: Rc<RefCell<Node>>, new_shape: Vec<usize>) -> Rc<RefCell<Node>> {
    let data = a.borrow().data.clone();

    let new_len: usize = new_shape.iter().product();
    assert_eq!(data.len(), new_len, "reshape: element count mismatch");

    let out = Node::new(data, new_shape);

    {
        let mut node = out.borrow_mut();
        node.parents = vec![a.clone()];
    }

    let a_clone = a.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
        let mut ig = a_clone.borrow_mut();
        for i in 0..grad.len() {
            ig.grad[i] += grad[i];
        }
    }));

    out
}