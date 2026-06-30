#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::{avgpool2d, layernorm, Embedding};
use neuralrs::nn::module::Module;
use neuralrs::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < tol, "{what} [{i}]: gpu {} cpu {}", a[i], b[i]);
    }
}

#[test]
fn cuda_layernorm() {
    let (rows, features) = (4usize, 6usize);
    let input: Vec<f32> = (0..rows * features).map(|i| (i % 13) as f32 * 0.2 - 1.0).collect();
    let gamma: Vec<f32> = (0..features).map(|i| 0.5 + (i % 3) as f32 * 0.3).collect();
    let beta: Vec<f32> = (0..features).map(|i| (i % 4) as f32 * 0.1 - 0.2).collect();
    let seed: Vec<f32> = (0..rows * features).map(|i| (i % 7) as f32 * 0.1 + 0.2).collect();
    let eps = 1e-5f32;

    let mut ln = neuralrs::nn::normalization::LayerNorm {
        gamma: Tensor::new(gamma.clone(), vec![features]),
        beta: Tensor::new(beta.clone(), vec![features]),
        epsilon: eps,
        num_features: features,
        gamma_grad: Rc::new(RefCell::new(vec![0.0; features])),
        beta_grad: Rc::new(RefCell::new(vec![0.0; features])),
    };
    let cx = Node::new(input.clone(), vec![rows, features]);
    let cout = ln.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    let cdin = cx.borrow().grad.clone();
    let cdg = ln.gamma_grad.borrow().clone();
    let cdb = ln.beta_grad.borrow().clone();

    let gx = Node::new(input, vec![rows, features]);
    let gg = Node::new(gamma, vec![features]);
    let gb = Node::new(beta, vec![features]);
    gpu::to_cuda(&gx);
    gpu::to_cuda(&gg);
    gpu::to_cuda(&gb);
    let gout = layernorm(&gx, &gg, &gb, eps);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gdin = gpu::read_grad(&gx);
    let gdg = gpu::read_grad(&gg);
    let gdb = gpu::read_grad(&gb);

    close(&gf, &cf, 1e-4, "layernorm fwd");
    close(&gdin, &cdin, 1e-4, "layernorm dinput");
    close(&gdg, &cdg, 1e-4, "layernorm dgamma");
    close(&gdb, &cdb, 1e-4, "layernorm dbeta");
    println!("layernorm: gpu matches cpu");
}

#[test]
fn cuda_avgpool() {
    let (n, c, h, w) = (2usize, 2usize, 4usize, 4usize);
    let (k, stride) = (2usize, 1usize);
    let input: Vec<f32> = (0..n * c * h * w).map(|i| (i % 17) as f32 * 0.1 - 0.5).collect();
    let out_h = (h - k) / stride + 1;
    let out_w = (w - k) / stride + 1;
    let seed: Vec<f32> = (0..n * c * out_h * out_w).map(|i| (i % 5) as f32 * 0.1 + 0.2).collect();

    let mut ap = neuralrs::nn::avgpool::AvgPool2d { kernel: k, stride, channels: c, in_h: h, in_w: w };
    let cx = Node::new(input.clone(), vec![n, c, h, w]);
    let cout = ap.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    let cdin = cx.borrow().grad.clone();

    let gx = Node::new(input, vec![n, c, h, w]);
    gpu::to_cuda(&gx);
    let gout = avgpool2d(&gx, k, stride);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gdin = gpu::read_grad(&gx);

    close(&gf, &cf, 1e-5, "avgpool fwd");
    close(&gdin, &cdin, 1e-5, "avgpool dinput");
    println!("avgpool: gpu matches cpu");
}

#[test]
fn cuda_embedding() {
    let (vocab, dim) = (10usize, 4usize);
    let weight: Vec<f32> = (0..vocab * dim).map(|i| (i % 19) as f32 * 0.1 - 0.9).collect();
    let indices = vec![3usize, 7, 1, 3, 0, 7];
    let seq_len = indices.len();
    let seed: Vec<f32> = (0..seq_len * dim).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let mut emb = neuralrs::nn::embedding::Embedding {
        weight: Tensor::new(weight.clone(), vec![vocab, dim]),
        vocab_size: vocab,
        embedding_dim: dim,
        weight_node: None,
    };
    let cout = emb.forward(&indices);
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    emb.sync_grads();
    let cdw = emb.weight.grad.clone();

    let gw = Node::new(weight, vec![vocab, dim]);
    gpu::to_cuda(&gw);
    let gemb = Embedding::new(gw.clone(), vocab, dim);
    let gout = gemb.forward(&indices);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gdw = gpu::read_grad(&gw);

    close(&gf, &cf, 1e-5, "embedding fwd");
    close(&gdw, &cdw, 1e-5, "embedding dweight");
    println!("embedding: gpu matches cpu");
}