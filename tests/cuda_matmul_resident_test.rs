#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::matmul;

#[test]
fn cuda_resident_matmul_backward() {
    let (m, k, n) = (40usize, 56usize, 32usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
    let seed: Vec<f32> = (0..m * n).map(|i| (i % 5) as f32 * 0.2 + 0.1).collect();

    let ca = Node::new(a.clone(), vec![m, k]);
    let cb = Node::new(b.clone(), vec![k, n]);
    let cc = cpu::matmul(ca.clone(), cb.clone());
    cc.borrow_mut().grad = seed.clone();
    backward_graph(&cc);
    let cc_data = cc.borrow().data.clone();
    let cga = ca.borrow().grad.clone();
    let cgb = cb.borrow().grad.clone();

    let ga = Node::new(a, vec![m, k]);
    let gb = Node::new(b, vec![k, n]);
    gpu::to_cuda(&ga);
    gpu::to_cuda(&gb);
    let gc = matmul(&ga, &gb);
    let gc_data = gpu::to_host(&gc);
    gpu::set_grad(&gc, &seed);
    backward_graph(&gc);
    let gga = gpu::read_grad(&ga);
    let ggb = gpu::read_grad(&gb);

    for i in 0..m * n {
        assert!((gc_data[i] - cc_data[i]).abs() < 1e-3, "fwd at {i}: gpu {} cpu {}", gc_data[i], cc_data[i]);
    }
    for i in 0..m * k {
        assert!((gga[i] - cga[i]).abs() < 1e-3, "a.grad at {i}: gpu {} cpu {}", gga[i], cga[i]);
    }
    for i in 0..k * n {
        assert!((ggb[i] - cgb[i]).abs() < 1e-3, "b.grad at {i}: gpu {} cpu {}", ggb[i], cgb[i]);
    }
    println!("resident matmul forward+backward: gpu matches cpu ({m}x{k} * {k}x{n})");
}