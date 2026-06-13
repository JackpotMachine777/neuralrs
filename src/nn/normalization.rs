use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub epsilon: f32,
    pub num_features: usize
}

impl Module for LayerNorm {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();
        let shape = input.borrow().shape.clone();
        let batch = shape[0];
        let features = shape[1];

        let mut out = vec![0.0; data.len()];

        for b in 0..batch {
            let row = &data[b*features..(b+1)*features];
            let mean = row.iter().sum::<f32>() / features as f32;
            let var = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / features as f32;

            for f in 0..features {
                let idx = b * features + f;
                let x_norm = (data[idx] - mean) / (var + self.epsilon).sqrt();
                out[idx] = self.gamma.data[f] * x_norm + self.beta.data[f];
            }
        }

        Node::new(out, shape)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![&mut self.gamma, &mut self.beta] }

    fn zero_grad(&mut self) { 
        self.gamma.grad = vec![0.0; self.gamma.data.len()];
        self.beta.grad = vec![0.0; self.beta.data.len()];
    }
}