#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::relu;

#[test]
fn cuda_resident_relu_backward() {
    let n: usize = 4096;
    let a: Vec<f32> = (0..n).map(|i| (i % 21) as f32 * 0.1 - 1.0).collect();
    let seed: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.3 + 0.1).collect();

    let ca = Node::new(a.clone(), vec![n]);
    let cout = cpu::relu(ca.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cout_data = cout.borrow().data.clone();
    let cga = ca.borrow().grad.clone();

    let ga = Node::new(a, vec![n]);
    gpu::to_cuda(&ga);
    let gout = relu(&ga);
    let gout_data = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gga = gpu::read_grad(&ga);

    for i in 0..n {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-4, "fwd at {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
        assert!((gga[i] - cga[i]).abs() < 1e-4, "a.grad at {i}: gpu {} cpu {}", gga[i], cga[i]);
    }
    println!("resident relu forward+backward: gpu matches cpu over {n} elements");
}