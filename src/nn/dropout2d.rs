use crate::tensor::Tensor;
use crate::nn::module::Module;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

/// Spatial dropout for conv feature maps: drops entire channels at once (instead
/// of individual pixels) during training.
///
/// Dropping whole channels works better than per-pixel dropout for
/// convolutional features, where neighbouring pixels are strongly correlated.
/// Does nothing in eval mode.
pub struct Dropout2d {
    pub probability: f32,
    pub training: bool,
}

impl Module for Dropout2d {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        if !self.training {
            return input.clone();
        }

        let shape = input.borrow().shape.clone();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let hw = h * w;
        let scale = 1.0 / (1.0 - self.probability);

        let mut channel_mask = vec![0.0; n * c];
        for i in 0..(n * c) {
            let r = rand::random::<f32>();
            channel_mask[i] = if r < self.probability { 0.0 } else { scale };
        }

        let in_data = input.borrow().data.clone();
        let mut out = vec![0.0; in_data.len()];
        for ni in 0..n {
            for ch in 0..c {
                let m = channel_mask[ni * c + ch];
                for p in 0..hw {
                    let idx = (ni * c + ch) * hw + p;
                    out[idx] = in_data[idx] * m;
                }
            }
        }

        let result = Node::new(out, shape.clone());
        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            let mut ig = input_clone.borrow_mut();
            for ni in 0..n {
                for ch in 0..c {
                    let m = channel_mask[ni * c + ch];
                    for p in 0..hw {
                        let idx = (ni * c + ch) * hw + p;
                        ig.grad[idx] += grad[idx] * m;
                    }
                }
            }
        }));

        result
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }

    fn zero_grad(&mut self) {}

    fn set_training(&mut self, training: bool) { self.training = training; }
}