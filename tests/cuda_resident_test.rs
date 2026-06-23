#![cfg(feature = "cuda")]

use neuralrs::autograd::node::Node;
use neuralrs::cuda::graph;

#[test]
fn cuda_resident_add_forward() {
    let n: usize = 5000;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.002).collect();
    let c: Vec<f32> = (0..n).map(|i| (i % 50) as f32 - 25.0).collect();

    let cpu: Vec<f32> = (0..n).map(|i| a[i] + b[i] + c[i]).collect();

    let ga = Node::new(a, vec![n]);
    let gb = Node::new(b, vec![n]);
    let gc = Node::new(c, vec![n]);
    
    graph::to_cuda(&ga);
    graph::to_cuda(&gb);
    graph::to_cuda(&gc);

    let ab = graph::add(&ga, &gb);
    let sum = graph::add(&ab, &gc);
    let out = graph::to_host(&sum);

    assert_eq!(out.len(), n);
    for i in 0..n {
        assert!(
            (out[i] - cpu[i]).abs() < 1e-4,
            "mismatch at {i}: gpu {} cpu {}",
            out[i],
            cpu[i]
        );
    }
    println!("resident (a+b)+c forward: gpu matches cpu over {n} elements");
}