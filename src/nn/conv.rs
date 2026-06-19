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
    pub padding: usize,
}

impl Module for Conv2d {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>> {
        let raw = input.borrow().data.clone();
        let in_shape = input.borrow().shape.clone();

        let n = in_shape[0];

        let c_in = self.c_in;
        let c_out = self.c_out;
        let kh = self.kh;
        let kw = self.kw;
        let stride = self.stride;
        let pad = self.padding;
        let in_h = self.in_h;
        let in_w = self.in_w;

        let ph = in_h + 2 * pad;
        let pw = in_w + 2 * pad;

        let mut data = vec![0.0; n * c_in * ph * pw];
        for ni in 0..n {
            for ic in 0..c_in {
                for y in 0..in_h {
                    for x in 0..in_w {
                        let src = ((ni * c_in + ic) * in_h + y) * in_w + x;
                        let dst = ((ni * c_in + ic) * ph + (y + pad)) * pw + (x + pad);
                        data[dst] = raw[src];
                    }
                }
            }
        }

        let out_h = (ph - kh) / stride + 1;
        let out_w = (pw - kw) / stride + 1;

        let weight = self.weight.storage.data.clone();
        let bias = self.bias.storage.data.clone();

        let mut out = vec![0.0; n * c_out * out_h * out_w];

        out.par_chunks_mut(out_h * out_w)
            .enumerate()
            .for_each(|(map_idx, out_map)| {
                let ni = map_idx / c_out;
                let oc = map_idx % c_out;
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = 0.0;
                        for ic in 0..c_in {
                            for i in 0..kh {
                                for j in 0..kw {
                                    let iy = oy * stride + i;
                                    let ix = ox * stride + j;
                                    let in_idx = ((ni * c_in + ic) * ph + iy) * pw + ix;
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

        let result = Node::new(out, vec![n, c_out, out_h, out_w]);

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
            let w_len = c_out * c_in * kh * kw;
            let units = n * c_out;

            let (w_grad, b_grad) = (0..units)
                .into_par_iter()
                .fold(
                    || (vec![0.0f32; w_len], vec![0.0f32; c_out]),
                    |(mut wg, mut bg), unit| {
                        let ni = unit / c_out;
                        let oc = unit % c_out;
                        for oy in 0..out_h {
                            for ox in 0..out_w {
                                let out_idx = ((ni * c_out + oc) * out_h + oy) * out_w + ox;
                                let g = grad[out_idx];
                                bg[oc] += g;

                                for ic in 0..c_in {
                                    for i in 0..kh {
                                        for j in 0..kw {
                                            let iy = oy * stride + i;
                                            let ix = ox * stride + j;
                                            let in_idx = ((ni * c_in + ic) * ph + iy) * pw + ix;
                                            let w_idx = ((oc * c_in + ic) * kh + i) * kw + j;
                                            wg[w_idx] += g * data_clone[in_idx];
                                        }
                                    }
                                }
                            }
                        }
                        (wg, bg)
                    },
                )
                .reduce(
                    || (vec![0.0f32; w_len], vec![0.0f32; c_out]),
                    |(mut wa, mut ba), (wb, bb)| {
                        for k in 0..w_len { wa[k] += wb[k]; }
                        for k in 0..c_out { ba[k] += bb[k]; }
                        (wa, ba)
                    },
                );

            let mut padded_in_grad = vec![0.0; n * c_in * ph * pw];
            let in_block = c_in * ph * pw;

            padded_in_grad
                .par_chunks_mut(in_block)
                .enumerate()
                .for_each(|(ni, in_chunk)| {
                    for oc in 0..c_out {
                        for oy in 0..out_h {
                            for ox in 0..out_w {
                                let out_idx = ((ni * c_out + oc) * out_h + oy) * out_w + ox;
                                let g = grad[out_idx];

                                for ic in 0..c_in {
                                    for i in 0..kh {
                                        for j in 0..kw {
                                            let iy = oy * stride + i;
                                            let ix = ox * stride + j;
                                            let in_local = (ic * ph + iy) * pw + ix;
                                            let w_idx = ((oc * c_in + ic) * kh + i) * kw + j;
                                            in_chunk[in_local] += g * weight_clone[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

            {
                let mut ig = input_clone.borrow_mut();
                for ni in 0..n {
                    for ic in 0..c_in {
                        for y in 0..in_h {
                            for x in 0..in_w {
                                let src = ((ni * c_in + ic) * ph + (y + pad)) * pw + (x + pad);
                                let dst = ((ni * c_in + ic) * in_h + y) * in_w + x;
                                ig.grad[dst] += padded_in_grad[src];
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