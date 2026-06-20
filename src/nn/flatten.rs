use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

/// Flattens everything except the batch dimension into one long vector per
/// sample: `[batch, c, h, w]` becomes `[batch, c*h*w]`.
///
/// Used to go from the conv part of a network into the dense part. It only
/// relabels the shape, so backward passes the gradient straight through.
pub struct Flatten { }

impl Module for Flatten {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();
        let shape =  input.borrow().shape.clone();

        let batch = shape[0];
        let features: usize = shape[1..].iter().product();

        let result = Node::new(data, vec![batch, features]);

        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            for i in 0..grad.len() {
                input_clone.borrow_mut().grad[i] += grad[i];
            }
        }));

        result
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }

    fn zero_grad(&mut self) {}
}