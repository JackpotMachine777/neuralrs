use crate::nn::module::Module;
use crate::tensor::Tensor;

pub struct RMSProp{
    pub lr: f32,
    pub beta: f32,
    pub epsilon: f32,
    pub v: Vec<f32>,
}

impl RMSProp{
    pub fn step_params(&mut self, params: &mut Vec<&mut Tensor>) {
        let mut idx = 0;
        for j in 0..params.len() {
            for k in 0..params[j].storage.data.len() {
                if self.v.len() <= idx { self.v.push(0.0); }

                self.v[idx] = self.beta * self.v[idx] + (1.0 - self.beta) * params[j].grad[k] * params[j].grad[k];
                params[j].storage.data[k] -= self.lr * params[j].grad[k] / (self.v[idx].sqrt() + self.epsilon);

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