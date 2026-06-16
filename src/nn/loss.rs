use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

pub trait Loss {
    fn forward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32;
    fn backward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>);
}

pub struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32 {
        let p = pred.borrow();
        let t = target.borrow();
        let n = p.data.len() as f32;
        p.data.iter().zip(t.data.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>() / n
    }

    fn backward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) {
        let grad: Vec<f32> = {
            let p = pred.borrow();
            let t = target.borrow();
            let n = p.data.len() as f32;
            (0..p.data.len())
                .map(|i| 2.0 * (p.data[i] - t.data[i]) / n)
                .collect()
        };

        pred.borrow_mut().grad = grad;
        pred.borrow_mut().backward();
    }
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32{
        let p = pred.borrow();
        let t = target.borrow();
        let eps = 1e-7;
        let n_samples = if p.shape.len() == 2 { p.shape[0] as f32 } else { 1.0 };
        let loss: f32 = p.data.iter().zip(t.data.iter())
            .map(|(pr, tg)| -tg * (pr + eps).ln())
            .sum::<f32>() / n_samples;
        
        loss
    }
    
    fn backward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) {
        let grad: Vec<f32> = {
            let p = pred.borrow();
            let t = target.borrow();
            let n_samples = if p.shape.len() == 2 { p.shape[0] as f32 } else { 1.0 };

            (0..p.data.len())
                .map(|i| (p.data[i] - t.data[i]) / n_samples)
                .collect()
        };

        pred.borrow_mut().grad = grad;
        pred.borrow_mut().backward();
    }
}