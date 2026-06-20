use crate::nn::module::Module;
use crate::tensor::Tensor;

pub struct Adagrad {
    pub lr: f32,
    pub epsilon: f32,
    pub g_sum: Vec<f32>,
    pub t: usize,
}

impl Adagrad {
    pub fn step_params(&mut self, params: &mut Vec<&mut Tensor>) {
        let mut idx = 0;
        for j in 0..params.len() {
            for k in 0..params[j].storage.data.len() {
                if self.t == 0 {
                    self.g_sum.push(0.0);
                }
                let g = params[j].grad[k];
                self.g_sum[idx] += g * g;
                params[j].storage.data[k] -= self.lr * g / (self.g_sum[idx].sqrt() + self.epsilon);
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