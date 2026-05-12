use crate::nn::module::Module;

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
    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>){
        let mut idx = 0;
        

        for i in 0..w.len(){
            let mut item = w[i].parameters();

            for j in 0..item.len(){
                for k in 0..item[j].data.len(){
                    if self.t == 0 {
                        self.m.push(0.0);
                        self.v.push(0.0);
                    }

                    self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * item[j].grad[k];
                    self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * item[j].grad[k] * item[j].grad[k];

                    let m_hat = self.m[idx] / (1.0 - self.beta1.powi(self.t as i32 + 1));
                    let v_hat = self.v[idx] / (1.0 - self.beta2.powi(self.t as i32 + 1));

                    item[j].data[k] = item[j].data[k] - self.lr * m_hat / (v_hat.sqrt() + self.epsilon);

                    idx += 1;
                }
            }
        }

        self.t += 1;
    }
}