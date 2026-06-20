use crate::nn::module::Module;
use crate::tensor::Tensor;

/// Stochastic gradient descent with momentum.
///
/// The classic optimizer: step each weight in the direction of its negative
/// gradient, with a `momentum` term that smooths the updates by carrying some of
/// the previous step's velocity forward.
pub struct SGD{
    pub lr: f32,
    pub momentum: f32,
    pub velocity: Vec<f32>,
}

impl SGD{
    pub fn step_params(&mut self, params: &mut Vec<&mut Tensor>) {
        let mut idx = 0;
        for j in 0..params.len() {
            for k in 0..params[j].storage.data.len() {
                if self.velocity.len() <= idx {
                    self.velocity.push(0.0);
                }
                self.velocity[idx] = self.momentum * self.velocity[idx] + params[j].grad[k];
                params[j].storage.data[k] -= self.lr * self.velocity[idx];
                idx += 1;
            }
        }
    }

    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>){
        let mut params: Vec<&mut Tensor> = Vec::new();
        for module in w.iter_mut() {
            params.extend(module.parameters());
        }
        self.step_params(&mut params);
    }
}