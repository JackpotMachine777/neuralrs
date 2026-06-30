#![cfg(feature = "cuda")]

use neuralrs::autograd::node::Node;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::optim::{clip_grad_norm, Adagrad, ADAM, NAdam, NesterovSGD, RMSProp, SGD};
use neuralrs::optim::{adagrad, adam, nadam, nesterov, rmsprop, sgd};
use neuralrs::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

type N = Rc<RefCell<Node>>;

fn run_gpu<F: FnMut(&[N])>(param: &[f32], grad: &[f32], steps: usize, mut step: F) -> Vec<f32> {
    let p = Node::new(param.to_vec(), vec![param.len()]);
    gpu::to_cuda(&p);
    for _ in 0..steps {
        gpu::set_grad(&p, grad);
        step(std::slice::from_ref(&p));
    }
    gpu::to_host(&p)
}

fn run_cpu<F: FnMut(&mut Vec<&mut Tensor>)>(param: &[f32], grad: &[f32], steps: usize, mut step: F) -> Vec<f32> {
    let mut t = Tensor::new(param.to_vec(), vec![param.len()]);
    for _ in 0..steps {
        t.grad = grad.to_vec();
        step(&mut vec![&mut t]);
    }
    t.storage.data.clone()
}

fn assert_close(g: &[f32], c: &[f32], tol: f32, name: &str) {
    for i in 0..g.len() {
        assert!((g[i] - c[i]).abs() < tol, "{name} param {i}: gpu {} cpu {}", g[i], c[i]);
    }
    println!("{name}: gpu matches cpu");
}

#[test]
fn cuda_optimizers() {
    let n = 256;
    let param: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let grad: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
    let s = 5;

    let mut g = SGD::new(0.01, 0.9);
    let mut c = sgd::SGD { lr: 0.01, momentum: 0.9, velocity: vec![] };
    assert_close(&run_gpu(&param, &grad, s, |p| g.step(p)), &run_cpu(&param, &grad, s, |t| c.step_params(t)), 1e-4, "sgd");

    let mut g = NesterovSGD::new(0.01, 0.9);
    let mut c = nesterov::NesterovSGD { lr: 0.01, momentum: 0.9, velocity: vec![], t: 0 };
    assert_close(&run_gpu(&param, &grad, s, |p| g.step(p)), &run_cpu(&param, &grad, s, |t| c.step_params(t)), 1e-4, "nesterov");

    let mut g = ADAM::new(0.01, 0.9, 0.999, 1e-8);
    let mut c = adam::ADAM { lr: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, t: 0, m: vec![], v: vec![] };
    assert_close(&run_gpu(&param, &grad, s, |p| g.step(p)), &run_cpu(&param, &grad, s, |t| c.step_params(t)), 1e-4, "adam");

    let mut g = NAdam::new(0.01, 0.9, 0.999, 1e-8);
    let mut c = nadam::NAdam { lr: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, t: 0, m: vec![], v: vec![] };
    assert_close(&run_gpu(&param, &grad, s, |p| g.step(p)), &run_cpu(&param, &grad, s, |t| c.step_params(t)), 1e-4, "nadam");

    let mut g = RMSProp::new(0.01, 0.99, 1e-8);
    let mut c = rmsprop::RMSProp { lr: 0.01, beta: 0.99, epsilon: 1e-8, v: vec![] };
    assert_close(&run_gpu(&param, &grad, s, |p| g.step(p)), &run_cpu(&param, &grad, s, |t| c.step_params(t)), 1e-4, "rmsprop");

    let mut g = Adagrad::new(0.1, 1e-8);
    let mut c = adagrad::Adagrad { lr: 0.1, epsilon: 1e-8, g_sum: vec![], t: 0 };
    assert_close(&run_gpu(&param, &grad, s, |p| g.step(p)), &run_cpu(&param, &grad, s, |t| c.step_params(t)), 1e-4, "adagrad");
}

#[test]
fn cuda_clip() {
    let n = 128;
    let g1: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.3 + 0.5).collect();
    let g2: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.4 - 0.2).collect();
    let max_norm = 1.0f32;

    let sumsq: f32 = g1.iter().chain(g2.iter()).map(|x| x * x).sum();
    let total = sumsq.sqrt();
    let scale = if total > max_norm { max_norm / total } else { 1.0 };
    let exp1: Vec<f32> = g1.iter().map(|x| x * scale).collect();
    let exp2: Vec<f32> = g2.iter().map(|x| x * scale).collect();

    let p1 = Node::new(vec![0.0; n], vec![n]);
    let p2 = Node::new(vec![0.0; n], vec![n]);
    gpu::to_cuda(&p1);
    gpu::to_cuda(&p2);
    gpu::set_grad(&p1, &g1);
    gpu::set_grad(&p2, &g2);
    clip_grad_norm(&[p1.clone(), p2.clone()], max_norm);
    let out1 = gpu::read_grad(&p1);
    let out2 = gpu::read_grad(&p2);

    for i in 0..n {
        assert!((out1[i] - exp1[i]).abs() < 1e-4, "clip p1 {i}: gpu {} exp {}", out1[i], exp1[i]);
        assert!((out2[i] - exp2[i]).abs() < 1e-4, "clip p2 {i}: gpu {} exp {}", out2[i], exp2[i]);
    }
    println!("clip: total norm {:.3} scaled by {:.4}, matches reference", total, scale);
}