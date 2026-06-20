use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::conv::Conv2d;
use rstorch::autograd::node::Node;
use rstorch::ops::im2col::conv2d_im2col;
use std::rc::Rc;
use std::cell::RefCell;

#[test]
fn im2col_matches_conv() {
    let n = 2;
    let c_in = 2;
    let c_out = 3;
    let in_h = 5;
    let in_w = 5;
    let kh = 3;
    let kw = 3;
    let stride = 1;
    let pad = 1;

    let w_len = c_out * c_in * kh * kw;
    let weight: Vec<f32> = (0..w_len).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let bias: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.5).collect();

    let in_len = n * c_in * in_h * in_w;
    let data: Vec<f32> = (0..in_len).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();

    let (out_im2col, oh, ow) = conv2d_im2col(
        &data, &weight, &bias,
        n, c_in, c_out, in_h, in_w, kh, kw, stride, pad,
    );

    let mut conv = Conv2d {
        weight: Tensor::new(weight.clone(), vec![c_out, c_in, kh, kw]),
        bias: Tensor::new(bias.clone(), vec![c_out]),
        c_in, c_out, kh, kw, stride, padding: pad, in_h, in_w,
        weight_grad: Rc::new(RefCell::new(vec![0.0; w_len])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; c_out])),
    };
    let input = Node::new(data.clone(), vec![n, c_in, in_h, in_w]);
    let out_conv = conv.forward(input);
    let out_conv_data = out_conv.borrow().data.clone();

    println!("im2col len: {}, conv len: {}", out_im2col.len(), out_conv_data.len());
    println!("out_h={oh}, out_w={ow}");

    assert_eq!(out_im2col.len(), out_conv_data.len());

    for i in 0..out_im2col.len() {
        let diff = (out_im2col[i] - out_conv_data[i]).abs();
        assert!(diff < 1e-4, "mismatch at {}: im2col={} conv={}", i, out_im2col[i], out_conv_data[i]);
    }

    println!("im2col matches conv ok");
}