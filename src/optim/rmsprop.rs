use crate::nn::module::Module;

pub struct RMSProp{
    pub lr: f32,
    pub beta: f32, 
    pub epsilon: f32,
    pub v: Vec<f32>,
}

impl RMSProp{
    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>){
        let mut idx = 0;

        for i in 0..w.len(){
            let mut item = w[i].parameters();

            for j in 0..item.len(){
                for k in 0..item[j].storage.data.len(){
                    if self.v.len() <= idx { self.v.push(0.0); }

                    self.v[idx] = self.beta * self.v[idx] + (1.0 - self.beta) * item[j].grad[k] * item[j].grad[k];
                    item[j].storage.data[k] = item[j].storage.data[k] - self.lr * item[j].grad[k] / (self.v[idx].sqrt() + self.epsilon);

                    idx += 1;
                }
            }
        }
    }
}