#![cfg(feature = "cuda")]

use neuralrs::autograd::node::Node;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::conv2d;
use neuralrs::ops::im2col::conv2d_im2col;
use neuralrs::autograd::node::backward_graph;
use neuralrs::nn::conv::Conv2d;
use neuralrs::nn::module::Module;
use neuralrs::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

#[test]
fn cuda_conv2d_forward() {
    let (n, c_in, in_h, in_w) = (2usize, 3, 8, 8);
    let (c_out, kh, kw, stride, pad) = (4usize, 3, 3, 1, 1);
    let in_len = n * c_in * in_h * in_w;
    let w_len = c_out * c_in * kh * kw;

    let input: Vec<f32> = (0..in_len).map(|i| (i % 23) as f32 * 0.05 - 0.5).collect();
    let weight: Vec<f32> = (0..w_len).map(|i| (i % 13) as f32 * 0.07 - 0.4).collect();
    let bias: Vec<f32> = (0..c_out).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();

    let (cpu_out, oh, ow) =
        conv2d_im2col(&input, &weight, &bias, n, c_in, c_out, in_h, in_w, kh, kw, stride, pad);

    let gi = Node::new(input, vec![n, c_in, in_h, in_w]);
    let gw = Node::new(weight, vec![c_out, c_in, kh, kw]);
    let gb = Node::new(bias, vec![c_out]);
    gpu::to_cuda(&gi);
    gpu::to_cuda(&gw);
    gpu::to_cuda(&gb);
    let gout = conv2d(&gi, &gw, &gb, stride, pad);
    let gpu_out = gpu::to_host(&gout);

    assert_eq!(gout.borrow().shape, vec![n, c_out, oh, ow]);
    assert_eq!(gpu_out.len(), cpu_out.len());
    for i in 0..cpu_out.len() {
        assert!((gpu_out[i] - cpu_out[i]).abs() < 1e-4, "conv at {i}: gpu {} cpu {}", gpu_out[i], cpu_out[i]);
    }
    println!("resident conv2d forward: gpu matches cpu (N={n} c_in={c_in} {in_h}x{in_w} -> c_out={c_out} {oh}x{ow}, k={kh}x{kw} s={stride} p={pad})");
}

#[test]
fn cuda_conv2d_backward() {
    let (n, c_in, in_h, in_w) = (2usize, 3, 7, 7);
    let (c_out, kh, kw, stride, pad) = (5usize, 3, 3, 1, 1);
    let in_len = n * c_in * in_h * in_w;
    let w_len = c_out * c_in * kh * kw;

    let input: Vec<f32> = (0..in_len).map(|i| (i % 23) as f32 * 0.05 - 0.5).collect();
    let weight: Vec<f32> = (0..w_len).map(|i| (i % 13) as f32 * 0.07 - 0.4).collect();
    let bias: Vec<f32> = (0..c_out).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();

    let mut layer = Conv2d {
        weight: Tensor::new(weight.clone(), vec![c_out, c_in, kh, kw]),
        bias: Tensor::new(bias.clone(), vec![c_out]),
        c_in, c_out, kh, kw, stride, padding: pad, in_h, in_w,
        weight_grad: Rc::new(RefCell::new(vec![0.0; w_len])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; c_out])),
    };
    let cin = Node::new(input.clone(), vec![n, c_in, in_h, in_w]);
    let cout = layer.forward(cin.clone());
    let out_len = cout.borrow().data.len();
    let seed: Vec<f32> = (0..out_len).map(|i| (i % 7) as f32 * 0.2 + 0.1).collect();
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let c_wgrad = layer.weight_grad.borrow().clone();
    let c_bgrad = layer.bias_grad.borrow().clone();
    let c_igrad = cin.borrow().grad.clone();

    let gi = Node::new(input, vec![n, c_in, in_h, in_w]);
    let gw = Node::new(weight, vec![c_out, c_in, kh, kw]);
    let gb = Node::new(bias, vec![c_out]);
    gpu::to_cuda(&gi);
    gpu::to_cuda(&gw);
    gpu::to_cuda(&gb);
    let gout = conv2d(&gi, &gw, &gb, stride, pad);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let g_igrad = gpu::read_grad(&gi);
    let g_wgrad = gpu::read_grad(&gw);
    let g_bgrad = gpu::read_grad(&gb);

    for i in 0..in_len {
        assert!((g_igrad[i] - c_igrad[i]).abs() < 1e-3, "input.grad at {i}: gpu {} cpu {}", g_igrad[i], c_igrad[i]);
    }
    for i in 0..w_len {
        assert!((g_wgrad[i] - c_wgrad[i]).abs() < 1e-3, "weight.grad at {i}: gpu {} cpu {}", g_wgrad[i], c_wgrad[i]);
    }
    for i in 0..c_out {
        assert!((g_bgrad[i] - c_bgrad[i]).abs() < 1e-3, "bias.grad at {i}: gpu {} cpu {}", g_bgrad[i], c_bgrad[i]);
    }
    println!("resident conv2d backward: gpu grads match cpu (input/weight/bias)");
}