use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct BatchNorm2d {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub epsilon: f32,
    pub num_channels: usize,
    pub gamma_grad: Rc<RefCell<Vec<f32>>>,
    pub beta_grad: Rc<RefCell<Vec<f32>>>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub momentum: f32,
    pub training: bool,
}

impl Module for BatchNorm2d {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();
        let shape = input.borrow().shape.clone();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let hw = h * w;
        let count = (n * hw) as f32;
        let eps = self.epsilon;

        let gamma = self.gamma.storage.data.clone();
        let beta = self.beta.storage.data.clone();

        let mut out = vec![0.0; data.len()];
        let mut x_norm = vec![0.0; data.len()];
        let mut inv_std = vec![0.0; c];

        if self.training {
            for ch in 0..c {
                let mut mean = 0.0;
                for ni in 0..n {
                    let base = (ni * c + ch) * hw;
                    for p in 0..hw {
                        mean += data[base + p];
                    }
                }
                mean /= count;

                let mut var = 0.0;
                for ni in 0..n {
                    let base = (ni * c + ch) * hw;
                    for p in 0..hw {
                        let d = data[base + p] - mean;
                        var += d * d;
                    }
                }
                var /= count;

                self.running_mean[ch] = self.momentum * self.running_mean[ch] + (1.0 - self.momentum) * mean;
                self.running_var[ch] = self.momentum * self.running_var[ch] + (1.0 - self.momentum) * var;

                let istd = 1.0 / (var + eps).sqrt();
                inv_std[ch] = istd;

                for ni in 0..n {
                    let base = (ni * c + ch) * hw;
                    for p in 0..hw {
                        let idx = base + p;
                        let xn = (data[idx] - mean) * istd;
                        x_norm[idx] = xn;
                        out[idx] = gamma[ch] * xn + beta[ch];
                    }
                }
            }
        } else {
            for ch in 0..c {
                let mean = self.running_mean[ch];
                let var = self.running_var[ch];
                let istd = 1.0 / (var + eps).sqrt();
                inv_std[ch] = istd;

                for ni in 0..n {
                    let base = (ni * c + ch) * hw;
                    for p in 0..hw {
                        let idx = base + p;
                        let xn = (data[idx] - mean) * istd;
                        x_norm[idx] = xn;
                        out[idx] = gamma[ch] * xn + beta[ch];
                    }
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
            let cnt = (n * hw) as f32;
            let mut dgamma = vec![0.0; c];
            let mut dbeta = vec![0.0; c];

            for ch in 0..c {
                let istd = inv_std[ch];

                if is_training {
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    for ni in 0..n {
                        let base = (ni * c + ch) * hw;
                        for p in 0..hw {
                            let idx = base + p;
                            let dxhat = grad[idx] * gamma[ch];
                            sum1 += dxhat;
                            sum2 += dxhat * x_norm[idx];
                        }
                    }
                    for ni in 0..n {
                        let base = (ni * c + ch) * hw;
                        for p in 0..hw {
                            let idx = base + p;
                            let dxhat = grad[idx] * gamma[ch];
                            let dx = istd * (dxhat - sum1 / cnt - x_norm[idx] * sum2 / cnt);
                            input_clone.borrow_mut().grad[idx] += dx;
                            dgamma[ch] += grad[idx] * x_norm[idx];
                            dbeta[ch] += grad[idx];
                        }
                    }
                } else {
                    for ni in 0..n {
                        let base = (ni * c + ch) * hw;
                        for p in 0..hw {
                            let idx = base + p;
                            let dx = grad[idx] * gamma[ch] * istd;
                            input_clone.borrow_mut().grad[idx] += dx;
                            dgamma[ch] += grad[idx] * x_norm[idx];
                            dbeta[ch] += grad[idx];
                        }
                    }
                }
            }

            let mut gg = gamma_grad_buf.borrow_mut();
            let mut bg = beta_grad_buf.borrow_mut();
            for ch in 0..c {
                gg[ch] += dgamma[ch];
                bg[ch] += dbeta[ch];
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