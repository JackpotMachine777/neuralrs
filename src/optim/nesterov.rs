use crate::nn::module::Module;

pub struct NesterovSGD {
    pub lr: f32,
    pub momentum: f32,
    pub velocity: Vec<f32>,
    pub t: usize,
}

impl NesterovSGD {
    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>) {
        let mut idx = 0;
        for i in 0..w.len() {
            let mut item = w[i].parameters();
            for j in 0..item.len() {
                for k in 0..item[j].storage.data.len() {
                    if self.t == 0 {
                        self.velocity.push(0.0);
                    }
                    let g = item[j].grad[k];
                    let v_prev = self.velocity[idx];
                    self.velocity[idx] = self.momentum * v_prev + g;
                    let update = self.momentum * self.velocity[idx] + g;
                    item[j].storage.data[k] -= self.lr * update;
                    idx += 1;
                }
            }
        }
        self.t += 1;
    }
}