//! Flatten layer: [N, ...] -> [N, rest], collapsing all but the batch dim.

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::node::Node;
use crate::cuda::graph::reshape;

/// Flattens `[N, ...]` to `[N, rest]`, keeps the batch dim, collapses the rest.
/// Thin wrapper over [`reshape`].
pub fn flatten(x: &Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
    let (n, rest) = {
        let xn = x.borrow();
        (xn.shape[0], xn.shape[1..].iter().product::<usize>())
    };
    reshape(x, vec![n, rest])
}