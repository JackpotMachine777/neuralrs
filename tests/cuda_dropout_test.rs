#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::dropout;

#[test]
fn cuda_dropout_training_mechanics() {
    let n = 10000usize;
    let p = 0.3f32;
    let scale = 1.0 / (1.0 - p);
    let input: Vec<f32> = (0..n).map(|i| ((i % 50) as f32 + 1.0) * 0.1).collect();
    let seed: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.2 + 0.5).collect();

    let gi = Node::new(input.clone(), vec![n]);
    gpu::to_cuda(&gi);
    let gout = dropout(&gi, p, true);
    let out = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let igrad = gpu::read_grad(&gi);

    let mut dropped = 0;
    for i in 0..n {
        if out[i] == 0.0 {
            dropped += 1;
            assert!(igrad[i].abs() < 1e-6, "dropped {i}: grad should be 0, got {}", igrad[i]);
        } else {
            assert!((out[i] - input[i] * scale).abs() < 1e-4, "kept {i}: out {} expected {}", out[i], input[i] * scale);
            assert!((igrad[i] - seed[i] * scale).abs() < 1e-4, "kept {i}: grad {} expected {}", igrad[i], seed[i] * scale);
        }
    }
    let frac = dropped as f32 / n as f32;
    assert!((frac - p).abs() < 0.05, "dropped fraction {frac} should be near {p}");
    println!("resident dropout (train): mask + grad routing correct, dropped {:.1}%", frac * 100.0);
}

#[test]
fn cuda_dropout_eval_passthrough() {
    let n = 256usize;
    let input: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.5).collect();
    let gi = Node::new(input.clone(), vec![n]);
    gpu::to_cuda(&gi);
    let gout = dropout(&gi, 0.5, false);
    let out = gpu::to_host(&gout);
    for i in 0..n {
        assert!((out[i] - input[i]).abs() < 1e-6, "eval passthrough {i}: {} vs {}", out[i], input[i]);
    }
    println!("resident dropout (eval): passthrough");
}