#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::reshape;

#[test]
fn cuda_resident_reshape_backward() {
    let (a, b) = (8usize, 12usize);
    let n = a * b;
    let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let seed: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.3 + 0.1).collect();

    let cx = Node::new(data.clone(), vec![a, b]);
    let cout = cpu::reshape(cx.clone(), vec![4, 24]);
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cout_data = cout.borrow().data.clone();
    let cgx = cx.borrow().grad.clone();

    let gx = Node::new(data, vec![a, b]);
    gpu::to_cuda(&gx);
    let gout = reshape(&gx, vec![4, 24]);
    let gout_data = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let ggx = gpu::read_grad(&gx);

    assert_eq!(gout.borrow().shape, vec![4, 24]);
    for i in 0..n {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-6, "fwd at {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
        assert!((ggx[i] - cgx[i]).abs() < 1e-6, "grad at {i}: gpu {} cpu {}", ggx[i], cgx[i]);
    }
    println!("resident reshape forward+backward: gpu matches cpu ([{a},{b}] -> [4,24])");
}