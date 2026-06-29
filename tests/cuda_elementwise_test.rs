#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::{abs, div, exp, log, sqrt, sub};
use std::cell::RefCell;
use std::rc::Rc;

type N = Rc<RefCell<Node>>;

fn check_unary(input: Vec<f32>, gpu_op: fn(&N) -> N, cpu_op: fn(N) -> N, tol: f32) {
    let n = input.len();
    let seed: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 + 0.3).collect();
    let cx = Node::new(input.clone(), vec![n]);
    let cout = cpu_op(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let c_out = cout.borrow().data.clone();
    let c_grad = cx.borrow().grad.clone();

    let gx = Node::new(input, vec![n]);
    gpu::to_cuda(&gx);
    let gout = gpu_op(&gx);
    let g_out = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let g_grad = gpu::read_grad(&gx);

    for i in 0..n {
        assert!((g_out[i] - c_out[i]).abs() < tol, "fwd {i}: gpu {} cpu {}", g_out[i], c_out[i]);
        assert!((g_grad[i] - c_grad[i]).abs() < tol, "grad {i}: gpu {} cpu {}", g_grad[i], c_grad[i]);
    }
}

fn check_binary(a: Vec<f32>, b: Vec<f32>, gpu_op: fn(&N, &N) -> N, cpu_op: fn(N, N) -> N, tol: f32) {
    let n = a.len();
    let seed: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 + 0.3).collect();
    let ca = Node::new(a.clone(), vec![n]);
    let cb = Node::new(b.clone(), vec![n]);
    let cout = cpu_op(ca.clone(), cb.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let c_out = cout.borrow().data.clone();
    let cga = ca.borrow().grad.clone();
    let cgb = cb.borrow().grad.clone();

    let ga = Node::new(a, vec![n]);
    let gb = Node::new(b, vec![n]);
    gpu::to_cuda(&ga);
    gpu::to_cuda(&gb);
    let gout = gpu_op(&ga, &gb);
    let g_out = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gga = gpu::read_grad(&ga);
    let ggb = gpu::read_grad(&gb);

    for i in 0..n {
        assert!((g_out[i] - c_out[i]).abs() < tol, "fwd {i}: gpu {} cpu {}", g_out[i], c_out[i]);
        assert!((gga[i] - cga[i]).abs() < tol, "a.grad {i}: gpu {} cpu {}", gga[i], cga[i]);
        assert!((ggb[i] - cgb[i]).abs() < tol, "b.grad {i}: gpu {} cpu {}", ggb[i], cgb[i]);
    }
}

#[test]
fn cuda_sub_div() {
    let n = 512;
    let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1 + 0.5).collect();
    check_binary(a.clone(), b.clone(), sub, cpu::sub, 1e-4);
    check_binary(a, b, div, cpu::div, 1e-3);
    println!("sub, div: gpu matches cpu");
}

#[test]
fn cuda_exp_log_sqrt_abs() {
    let n = 512;
    let moderate: Vec<f32> = (0..n).map(|i| (i % 20) as f32 * 0.05 - 0.5).collect();
    let positive: Vec<f32> = (0..n).map(|i| (i % 20) as f32 * 0.1 + 0.5).collect();
    let signed: Vec<f32> = (0..n).map(|i| (i % 21) as f32 * 0.1 - 1.0).collect();
    check_unary(moderate, exp, cpu::exp, 1e-4);
    check_unary(positive.clone(), log, cpu::log, 1e-4);
    check_unary(positive, sqrt, cpu::sqrt, 1e-4);
    check_unary(signed, abs, cpu::abs, 1e-4);
    println!("exp, log, sqrt, abs: gpu matches cpu");
}