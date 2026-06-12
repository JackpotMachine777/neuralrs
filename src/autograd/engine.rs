use std::{rc::Rc, cell::RefCell};
use crate::autograd::node::Node;

pub fn backward(node: Rc<RefCell<Node>>){
    let len = node.borrow().data.len();
    node.borrow_mut().grad = vec![1.0; len];
    node.borrow_mut().backward();
}