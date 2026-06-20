use crate::tensor::Tensor;
use crate::nn::module::Module;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

/// Dropout: during training, randomly zeroes out each value with probability
/// `probability`, and scales the survivors up to keep the overall magnitude
/// roughly the same.
///
/// In eval mode it does nothing (passes the input through). Helps prevent
/// overfitting by stopping the network from leaning too hard on any one
/// activation. The `mask` records which values survived so backward can route
/// gradients the same way.
pub struct Dropout {
    pub probability: f32,
    pub mask: Vec<f32>,
    pub training: bool,
}

impl Module for Dropout {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        if !self.training {
            return input.clone();
        }

        let scale = 1.0 / (1.0 - self.probability);
        let in_data = input.borrow().data.clone();

        let mut mask = vec![0.0; in_data.len()];
        let mut out = vec![0.0; in_data.len()];
        for i in 0..in_data.len() {
            let r = rand::random::<f32>();
            if r < self.probability {
                mask[i] = 0.0;
                out[i] = 0.0;
            } else {
                mask[i] = scale;
                out[i] = in_data[i] * scale;
            }
        }

        let shape = input.borrow().shape.clone();
        let result = Node::new(out, shape);

        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            let mut ig = input_clone.borrow_mut();
            for i in 0..grad.len() {
                ig.grad[i] += grad[i] * mask[i];
            }
        }));

        result
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn zero_grad(&mut self) {}
    fn set_training(&mut self, training: bool) { self.training = training; }
}