use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct BatchNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub epsilon: f32,
    pub num_features: usize,
    pub gamma_grad: Rc<RefCell<Vec<f32>>>,
    pub beta_grad: Rc<RefCell<Vec<f32>>>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub momentum: f32,
    pub training: bool,
}

impl Module for BatchNorm {
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
        let mut inv_std = vec![0.0; features];

        if self.training {
            for f in 0..features {
                let mut mean = 0.0;
                for b in 0..batch {
                    mean += data[b * features + f];
                }
                mean /= batch as f32;

                let mut var = 0.0;
                for b in 0..batch {
                    let d = data[b * features + f] - mean;
                    var += d * d;
                }
                var /= batch as f32;

                self.running_mean[f] = self.momentum * self.running_mean[f] + (1.0 - self.momentum) * mean;
                self.running_var[f] = self.momentum * self.running_var[f] + (1.0 - self.momentum) * var;

                let istd = 1.0 / (var + eps).sqrt();
                inv_std[f] = istd;

                for b in 0..batch {
                    let idx = b * features + f;
                    let xn = (data[idx] - mean) * istd;
                    x_norm[idx] = xn;
                    out[idx] = gamma[f] * xn + beta[f];
                }
            }
        } else {
            for f in 0..features {
                let mean = self.running_mean[f];
                let var = self.running_var[f];
                let istd = 1.0 / (var + eps).sqrt();
                inv_std[f] = istd;

                for b in 0..batch {
                    let idx = b * features + f;
                    let xn = (data[idx] - mean) * istd;
                    x_norm[idx] = xn;
                    out[idx] = gamma[f] * xn + beta[f];
                }
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
        let is_training = self.training;

        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            let n = batch as f32;
            let mut dgamma = vec![0.0; features];
            let mut dbeta = vec![0.0; features];

            for f in 0..features {
                let istd = inv_std[f];

                if is_training {
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    for b in 0..batch {
                        let idx = b * features + f;
                        let dxhat = grad[idx] * gamma[f];
                        sum1 += dxhat;
                        sum2 += dxhat * x_norm[idx];
                    }
                    for b in 0..batch {
                        let idx = b * features + f;
                        let dxhat = grad[idx] * gamma[f];
                        let dx = istd * (dxhat - sum1 / n - x_norm[idx] * sum2 / n);
                        input_clone.borrow_mut().grad[idx] += dx;
                        dgamma[f] += grad[idx] * x_norm[idx];
                        dbeta[f] += grad[idx];
                    }
                } else {
                    for b in 0..batch {
                        let idx = b * features + f;
                        let dx = grad[idx] * gamma[f] * istd;
                        input_clone.borrow_mut().grad[idx] += dx;
                        dgamma[f] += grad[idx] * x_norm[idx];
                        dbeta[f] += grad[idx];
                    }
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

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}