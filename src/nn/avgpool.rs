use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

/// Average pooling over `kernel × kernel` windows, taking the mean of each.
///
/// Like [`MaxPool2d`] but averages instead of taking the max. Backward spreads
/// each output's gradient evenly across all the inputs that fed into that
/// window.
///
/// [`MaxPool2d`]: crate::nn::maxpool::MaxPool2d
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
        let in_shape = input.borrow().shape.clone();
        let n = in_shape[0];

        let c = self.channels;
        let k = self.kernel;
        let stride = self.stride;
        let in_h = self.in_h;
        let in_w = self.in_w;

        let out_h = (in_h - k) / stride + 1;
        let out_w = (in_w - k) / stride + 1;
        let area = (k * k) as f32;

        let mut out = vec![0.0; n * c * out_h * out_w];

        for ni in 0..n {
            for ch in 0..c {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = 0.0;
                        for i in 0..k {
                            for j in 0..k {
                                let iy = oy * stride + i;
                                let ix = ox * stride + j;
                                let in_idx = ((ni * c + ch) * in_h + iy) * in_w + ix;
                                sum += data[in_idx];
                            }
                        }
                        let out_idx = ((ni * c + ch) * out_h + oy) * out_w + ox;
                        out[out_idx] = sum / area;
                    }
                }
            }
        }

        let result = Node::new(out, vec![n, c, out_h, out_w]);

        {
            let mut node = result.borrow_mut();
            node.parents = vec![input.clone()];
        }

        let input_clone = input.clone();
        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            for ni in 0..n {
                for ch in 0..c {
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let out_idx = ((ni * c + ch) * out_h + oy) * out_w + ox;
                            let g = grad[out_idx] / area;
                            for i in 0..k {
                                for j in 0..k {
                                    let iy = oy * stride + i;
                                    let ix = ox * stride + j;
                                    let in_idx = ((ni * c + ch) * in_h + iy) * in_w + ix;
                                    input_clone.borrow_mut().grad[in_idx] += g;
                                }
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