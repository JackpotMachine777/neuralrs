use crate::nn::module::Module;

pub struct SGD{
    pub lr: f32,
    pub momentum: f32,
    pub velocity: Vec<f32>,
}

impl SGD{
    pub fn step(&mut self, w: &mut Vec<Box<dyn Module>>){
        let mut idx = 0;

        for i in 0..w.len(){
            let mut item = w[i].parameters();

            for j in 0..item.len(){
                for k in 0..item[j].data.len(){
                    if self.velocity.len() <= idx{
                        self.velocity.push(0.0);
                    }

                    self.velocity[idx] = self.momentum * self.velocity[idx]  + item[j].grad[k];
                    item[j].data[k] = item[j].data[k] - self.lr * self.velocity[idx];

                    idx += 1;
                }
            }
        }
    }
}