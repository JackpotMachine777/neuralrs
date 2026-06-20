use crate::nn::module::Module;
use crate::tensor::Tensor;

pub struct ADAM{
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ADAM{
    pub fn step_params(&mut self, params: &mut Vec<&mut Tensor>) {
        let mut idx = 0;
        for j in 0..params.len() {
            for k in 0..params[j].storage.data.len() {
                if self.t == 0 {
                    self.m.push(0.0);
                    self.v.push(0.0);
                }

                self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * params[j].grad[k];
                self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * params[j].grad[k] * params[j].grad[k];

                let m_hat = self.m[idx] / (1.0 - self.beta1.powi(self.t as i32 + 1));
                let v_hat = self.v[idx] / (1.0 - self.beta2.powi(self.t as i32 + 1));

                params[j].storage.data[k] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);

                idx += 1;
            }
        }
        self.t += 1;
    }

    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>){
        let mut params: Vec<&mut Tensor> = Vec::new();
        for module in w.iter_mut() {
            params.extend(module.parameters());
        }
        self.step_params(&mut params);
    }
}