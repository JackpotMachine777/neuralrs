use crate::nn::module::Module;

pub struct SGD{
    pub lr: f32,
}

impl SGD{
    pub fn step(&self, w: &mut Vec<Box<dyn Module>>){
        for i in 0..w.len(){
            let mut item = w[i].parameters();

            for j in 0..item.len(){
                for k in 0..item[j].data.len(){
                    item[j].data[k] = item[j].data[k] - self.lr * item[j].grad[k];
                }
            }
        }
    }
}