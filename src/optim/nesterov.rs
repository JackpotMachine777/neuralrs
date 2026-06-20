use crate::nn::module::Module;
use crate::tensor::Tensor;

/// SGD with Nesterov momentum.
///
/// A momentum variant that "looks ahead" — it computes the gradient at the point
/// momentum is about to carry the weights to, which often converges a bit
/// cleaner than plain momentum.
pub struct NesterovSGD {
    pub lr: f32,
    pub momentum: f32,
    pub velocity: Vec<f32>,
    pub t: usize,
}

impl NesterovSGD {
    pub fn step_params(&mut self, params: &mut Vec<&mut Tensor>) {
        let mut idx = 0;
        for j in 0..params.len() {
            for k in 0..params[j].storage.data.len() {
                if self.t == 0 {
                    self.velocity.push(0.0);
                }
                let g = params[j].grad[k];
                let v_prev = self.velocity[idx];
                self.velocity[idx] = self.momentum * v_prev + g;
                let update = self.momentum * self.velocity[idx] + g;
                params[j].storage.data[k] -= self.lr * update;
                idx += 1;
            }
        }
        self.t += 1;
    }

    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>) {
        let mut params: Vec<&mut Tensor> = Vec::new();
        for module in w.iter_mut() {
            params.extend(module.parameters());
        }
        self.step_params(&mut params);
    }
}