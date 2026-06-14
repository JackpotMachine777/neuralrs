use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;
use rayon::prelude::*;

pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub c_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub weight_grad: Rc<RefCell<Vec<f32>>>,
    pub bias_grad: Rc<RefCell<Vec<f32>>>,
}

impl Module for Conv2d {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let data = input.borrow().data.clone();

        let c_in = self.c_in;
        let c_out = self.c_out;
        let kh = self.kh;
        let kw = self.kw;
        let stride = self.stride;
        let in_h = self.in_h;
        let in_w = self.in_w;

        let out_h = (in_h - kh) / stride + 1;
        let out_w = (in_w - kw) / stride + 1;

        let weight = self.weight.storage.data.clone();
        let bias = self.bias.storage.data.clone();

        let mut out = vec![0.0; c_out * out_h * out_w];

        out.par_chunks_mut(out_h * out_w)
            .enumerate()
            .for_each(|(oc, out_map)| {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = 0.0;
                        for ic in 0..c_in {
                            for i in 0..kh {
                                for j in 0..kw {
                                    let iy = oy * stride + i;
                                    let ix = ox * stride + j;
                                    let in_idx = (ic * in_h + iy) * in_w + ix;
                                    let w_idx = ((oc * c_in + ic) * kh + i) * kw + j;
                                    sum += data[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        sum += bias[oc];
                        out_map[oy * out_w + ox] = sum;
                    }
                }
            });

        let result = Node::new(out, vec![c_out, out_h, out_w]);

        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        let weight_clone = weight.clone();
        let data_clone = data.clone();
        let weight_grad_buf = self.weight_grad.clone();
        let bias_grad_buf = self.bias_grad.clone();

        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            let mut w_grad = vec![0.0; c_out * c_in * kh * kw];
            let mut b_grad = vec![0.0; c_out];

            for oc in 0..c_out {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let out_idx = (oc * out_h + oy) * out_w + ox;
                        let g = grad[out_idx];

                        b_grad[oc] += g;

                        for ic in 0..c_in {
                            for i in 0..kh {
                                for j in 0..kw {
                                    let iy = oy * stride + i;
                                    let ix = ox * stride + j;
                                    let in_idx = (ic * in_h + iy) * in_w + ix;
                                    let w_idx = ((oc * c_in + ic) * kh + i) * kw + j;

                                    w_grad[w_idx] += g * data_clone[in_idx];
                                    input_clone.borrow_mut().grad[in_idx] += g * weight_clone[w_idx];
                                }
                            }
                        }
                    }
                }
            }

            let mut wg = weight_grad_buf.borrow_mut();
            let mut bg = bias_grad_buf.borrow_mut();

            for k in 0..w_grad.len() { wg[k] += w_grad[k]; }
            for k in 0..b_grad.len() { bg[k] += b_grad[k]; }
        }));

        result
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn zero_grad(&mut self) {
        self.weight.grad = vec![0.0; self.weight.storage.data.len()];
        self.bias.grad = vec![0.0; self.bias.storage.data.len()];
        *self.weight_grad.borrow_mut() = vec![0.0; self.weight.storage.data.len()];
        *self.bias_grad.borrow_mut() = vec![0.0; self.bias.storage.data.len()];
    }

    fn sync_grads(&mut self) {
        self.weight.grad = self.weight_grad.borrow().clone();
        self.bias.grad = self.bias_grad.borrow().clone();
    }
}