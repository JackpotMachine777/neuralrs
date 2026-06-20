use crate::nn::module::Module;
use crate::tensor::Tensor;

/// NAdam optimizer — Adam with Nesterov-style momentum.
///
/// Combines Adam's per-weight adaptive steps with the look-ahead idea from
/// Nesterov momentum.
pub struct NAdam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl NAdam {
    pub fn step_params(&mut self, params: &mut Vec<&mut Tensor>) {
        let mut idx = 0;
        let t = self.t as i32 + 1;
        let b1 = self.beta1;
        let b2 = self.beta2;

        for j in 0..params.len() {
            for k in 0..params[j].storage.data.len() {
                if self.t == 0 {
                    self.m.push(0.0);
                    self.v.push(0.0);
                }
                let g = params[j].grad[k];

                self.m[idx] = b1 * self.m[idx] + (1.0 - b1) * g;
                self.v[idx] = b2 * self.v[idx] + (1.0 - b2) * g * g;

                let m_hat = self.m[idx] / (1.0 - b1.powi(t));
                let v_hat = self.v[idx] / (1.0 - b2.powi(t));

                let m_nesterov = b1 * m_hat + (1.0 - b1) * g / (1.0 - b1.powi(t));

                params[j].storage.data[k] -= self.lr * m_nesterov / (v_hat.sqrt() + self.epsilon);
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