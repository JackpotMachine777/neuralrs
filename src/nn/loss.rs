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
    fn forward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32 {
        let p = pred.borrow();
        let t = target.borrow();

        let (batch, classes) = if p.shape.len() == 2 {
            (p.shape[0], p.shape[1])
        } else {
            (1, p.shape[0])
        };

        let mut total = 0.0;
        for b in 0..batch {
            let start = b * classes;
            let row = &p.data[start..start + classes];

            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            let log_sum_exp = sum_exp.ln() + max;

            for c in 0..classes {
                let log_softmax = row[c] - log_sum_exp;
                total += -t.data[start + c] * log_softmax;
            }
        }

        total / batch as f32
    }

    fn backward(&self, pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) {
        let grad: Vec<f32> = {
            let p = pred.borrow();
            let t = target.borrow();

            let (batch, classes) = if p.shape.len() == 2 {
                (p.shape[0], p.shape[1])
            } else {
                (1, p.shape[0])
            };

            let mut g = vec![0.0; p.data.len()];
            for b in 0..batch {
                let start = b * classes;
                let row = &p.data[start..start + classes];

                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();

                for c in 0..classes {
                    let softmax_c = exps[c] / sum;
                    g[start + c] = (softmax_c - t.data[start + c]) / batch as f32;
                }
            }
            g
        };
        pred.borrow_mut().grad = grad;
        pred.borrow_mut().backward();
    }
}