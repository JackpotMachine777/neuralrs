use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub epsilon: f32,
    pub num_features: usize,
    pub gamma_grad: Rc<RefCell<Vec<f32>>>,
    pub beta_grad: Rc<RefCell<Vec<f32>>>,
}

impl Module for LayerNorm {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();
        let shape = input.borrow().shape.clone();
        let batch = shape[0];
        let features = shape[1];
        let eps = self.epsilon;

        let gamma = self.gamma.storage.data.clone();
        let beta = self.beta.storage.data.clone();

        let mut out = vec![0.0; data.len()];
        let mut x_norm = vec![0.0; data.len()];
        let mut inv_std = vec![0.0; batch];

        for b in 0..batch {
            let start = b * features;
            let row = &data[start..start + features];
            let mean = row.iter().sum::<f32>() / features as f32;
            let var = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / features as f32;
            let istd = 1.0 / (var + eps).sqrt();
            inv_std[b] = istd;

            for f in 0..features {
                let idx = start + f;
                let xn = (data[idx] - mean) * istd;
                x_norm[idx] = xn;
                out[idx] = gamma[f] * xn + beta[f];
            }
        }

        let result = Node::new(out, shape.clone());

        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        let gamma_grad_buf = self.gamma_grad.clone();
        let beta_grad_buf = self.beta_grad.clone();

        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            let mut dgamma = vec![0.0; features];
            let mut dbeta = vec![0.0; features];

            for b in 0..batch {
                let start = b * features;
                let istd = inv_std[b];

                let mut sum1 = 0.0;
                let mut sum2 = 0.0;

                for f in 0..features {
                    let idx = start + f;
                    let dxhat = grad[idx] * gamma[f];
                    sum1 += dxhat;
                    sum2 += dxhat * x_norm[idx];
                }

                let n = features as f32;

                for f in 0..features {
                    let idx = start + f;
                    let dxhat = grad[idx] * gamma[f];

                    let dx = istd * (dxhat - sum1 / n - x_norm[idx] * sum2 / n);
                    input_clone.borrow_mut().grad[idx] += dx;

                    dgamma[f] += grad[idx] * x_norm[idx];
                    dbeta[f] += grad[idx];
                }
            }

            let mut gg = gamma_grad_buf.borrow_mut();
            let mut bg = beta_grad_buf.borrow_mut();

            for f in 0..features {
                gg[f] += dgamma[f];
                bg[f] += dbeta[f];
            }
        }));

        result
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![&mut self.gamma, &mut self.beta] }

    fn zero_grad(&mut self) { 
        self.gamma.grad = vec![0.0; self.gamma.storage.data.len()];
        self.beta.grad = vec![0.0; self.beta.storage.data.len()];

        *self.gamma_grad.borrow_mut() = vec![0.0; self.gamma.storage.data.len()];
        *self.beta_grad.borrow_mut() = vec![0.0; self.beta.storage.data.len()];
    }

    fn sync_grads(&mut self) {
        self.gamma.grad = self.gamma_grad.borrow().clone();
        self.beta.grad = self.beta_grad.borrow().clone();
    }
}