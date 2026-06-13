use crate::tensor::Tensor;
use crate::nn::module::Module;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

pub struct Dropout{
    pub probability: f32,
    pub mask: Vec<f32>,
    pub training: bool,
}

impl Module for Dropout{
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        if self.training == false { return input.clone(); }

        let mut out = Vec::with_capacity(input.borrow().data.len());
        self.mask.clear();

        for &x in &input.borrow().data {
            let n = rand::random::<f32>();

            if n < self.probability { 
                self.mask.push(0.0);
                out.push(0.0); 
            }
            else { 
                self.mask.push(1.0);
                out.push(x * (1.0 / (1.0 - self.probability))); 
            }
        }

        let shape = input.borrow().shape.clone();
        Node::new(out, shape)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }

    fn zero_grad(&mut self) { }
}