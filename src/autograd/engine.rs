use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

/// Standalone recursive backward entry point (older helper). The main backward
/// logic now lives on [`Node`] in the `node` module.
///
/// [`Node`]: crate::autograd::node::Node
pub fn backward(node: Rc<RefCell<Node>>){
    let len = node.borrow().data.len();
    node.borrow_mut().grad = vec![1.0; len];
    node.borrow_mut().backward();
}