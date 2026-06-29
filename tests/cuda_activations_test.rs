#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::{elu, gelu, leaky_relu, pow, scale, sigmoid, silu, tanh};
use std::cell::RefCell;
use std::rc::Rc;

type N = Rc<RefCell<Node>>;

fn check_unary(input: Vec<f32>, gpu_op: fn(&N) -> N, cpu_op: fn(N) -> N, tol: f32, name: &str) {
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
        assert!((g_out[i] - c_out[i]).abs() < tol, "{name} fwd {i}: gpu {} cpu {}", g_out[i], c_out[i]);
        assert!((g_grad[i] - c_grad[i]).abs() < tol, "{name} grad {i}: gpu {} cpu {}", g_grad[i], c_grad[i]);
    }
    println!("{name}: gpu matches cpu");
}

#[test]
fn cuda_activations() {
    let n = 512;
    let wide: Vec<f32> = (0..n).map(|i| (i % 41) as f32 * 0.15 - 3.0).collect();
    let positive: Vec<f32> = (0..n).map(|i| (i % 30) as f32 * 0.1 + 0.3).collect();

    check_unary(wide.clone(), sigmoid, cpu::sigmoid, 1e-4, "sigmoid");
    check_unary(wide.clone(), tanh, cpu::tanh, 1e-4, "tanh");
    check_unary(wide.clone(), gelu, cpu::gelu, 1e-3, "gelu");
    check_unary(wide.clone(), silu, cpu::silu, 1e-4, "silu");
    check_unary(wide.clone(), |x| elu(x, 1.0), |x| cpu::elu(x, 1.0), 1e-4, "elu");
    check_unary(wide.clone(), |x| leaky_relu(x, 0.01), |x| cpu::leaky_relu(x, 0.01), 1e-4, "leaky_relu");
    check_unary(wide, |x| scale(x, 2.5), |x| cpu::scale(x, 2.5), 1e-4, "scale");
    check_unary(positive, |x| pow(x, 2.5), |x| cpu::pow(x, 2.5), 1e-3, "pow");
}