#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::maxpool2d;
use neuralrs::nn::maxpool::MaxPool2d;
use neuralrs::nn::module::Module;

fn check(n: usize, c: usize, in_h: usize, in_w: usize, k: usize, stride: usize) {
    let in_len = n * c * in_h * in_w;
    let input: Vec<f32> = (0..in_len).map(|i| ((i * 31 + 7) % 97) as f32 * 0.05 - 2.0).collect();

    let mut layer = MaxPool2d { kernel: k, stride, channels: c, in_h, in_w };
    let cin = Node::new(input.clone(), vec![n, c, in_h, in_w]);
    let cout = layer.forward(cin.clone());
    let out_len = cout.borrow().data.len();
    let cout_data = cout.borrow().data.clone();
    let seed: Vec<f32> = (0..out_len).map(|i| (i % 7) as f32 * 0.2 + 0.1).collect();
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let c_igrad = cin.borrow().grad.clone();

    let gi = Node::new(input, vec![n, c, in_h, in_w]);
    gpu::to_cuda(&gi);
    let gout = maxpool2d(&gi, k, stride);
    let gout_data = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let g_igrad = gpu::read_grad(&gi);

    assert_eq!(gout.borrow().shape, cout.borrow().shape);
    for i in 0..out_len {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-5, "fwd at {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
    }
    for i in 0..in_len {
        assert!((g_igrad[i] - c_igrad[i]).abs() < 1e-4, "input.grad at {i}: gpu {} cpu {}", g_igrad[i], c_igrad[i]);
    }
}

#[test]
fn cuda_maxpool_non_overlapping() {
    check(2, 3, 8, 8, 2, 2);
    println!("resident maxpool (k=2 s=2): gpu matches cpu");
}

#[test]
fn cuda_maxpool_overlapping() {
    check(2, 3, 7, 7, 3, 1);
    println!("resident maxpool (k=3 s=1, overlapping): gpu matches cpu");
}