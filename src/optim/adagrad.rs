use crate::nn::module::Module;

pub struct Adagrad {
    pub lr: f32,
    pub epsilon: f32,
    pub g_sum: Vec<f32>,
    pub t: usize,
}

impl Adagrad {
    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>) {
        let mut idx = 0;
        for i in 0..w.len() {
            let mut item = w[i].parameters();
            for j in 0..item.len() {
                for k in 0..item[j].storage.data.len() {
                    if self.t == 0 {
                        self.g_sum.push(0.0);
                    }
                    let g = item[j].grad[k];
                    self.g_sum[idx] += g * g;
                    item[j].storage.data[k] -= self.lr * g / (self.g_sum[idx].sqrt() + self.epsilon);
                    idx += 1;
                }
            }
        }
        self.t += 1;
    }
}