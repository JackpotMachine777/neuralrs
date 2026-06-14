use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct AvgPool2d {
    pub kernel: usize,
    pub stride: usize,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
}


impl Module for AvgPool2d {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();
        let c = self.channels;
        let k = self.kernel;
        let stride = self.stride;
        let in_h = self.in_h;
        let in_w = self.in_w;

        let out_h = (in_h - k) / stride + 1;
        let out_w = (in_w - k) / stride + 1;
        let area = (k * k) as f32;

        let mut out = vec![0.0; c * out_h * out_w];

        for ch in 0..c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = 0.0;
                    for i in 0..k {
                        for j in 0..k {
                            let iy = oy * stride + i;
                            let ix = ox * stride + j;
                            let in_idx = (ch * in_h + iy) * in_w + ix;
                            sum += data[in_idx];
                        }
                    }
                    let out_idx = (ch * out_h + oy) * out_w + ox;
                    out[out_idx] = sum / area;
                }
            }
        }

        let result = Node::new(out, vec![c, out_h, out_w]);

        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            for ch in 0..c {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let out_idx = (ch * out_h + oy) * out_w + ox;
                        let g = grad[out_idx] / area;
                        for i in 0..k {
                            for j in 0..k {
                                let iy = oy * stride + i;
                                let ix = ox * stride + j;
                                let in_idx = (ch * in_h + iy) * in_w + ix;
                                input_clone.borrow_mut().grad[in_idx] += g;
                            }
                        }
                    }
                }
            }
        }));

        result
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn zero_grad(&mut self) {}
}