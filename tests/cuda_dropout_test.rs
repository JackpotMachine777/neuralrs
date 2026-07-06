#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::{dropout, dropout2d};

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

#[test]
fn cuda_dropout2d_channel_mechanics() {
    let (n, c, h, w) = (8usize, 16usize, 7usize, 7usize);
    let hw = h * w;
    let total = n * c * hw;
    let p = 0.4f32;
    let scale = 1.0 / (1.0 - p);
    let input: Vec<f32> = (0..total).map(|i| ((i % 50) as f32 + 1.0) * 0.1).collect();
    let seed: Vec<f32> = (0..total).map(|i| (i % 7) as f32 * 0.2 + 0.5).collect();

    let gi = Node::new(input.clone(), vec![n, c, h, w]);
    gpu::to_cuda(&gi);
    let gout = dropout2d(&gi, p, true);
    let out = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let igrad = gpu::read_grad(&gi);

    let mut dropped = 0;
    for slot in 0..(n * c) {
        let base = slot * hw;
        if out[base] == 0.0 {
            dropped += 1;
            for q in 0..hw {
                assert!(out[base + q] == 0.0, "channel {slot}: pixel {q} survived a dropped channel");
                assert!(igrad[base + q].abs() < 1e-6, "channel {slot}: grad leaked at {q}");
            }
        } else {
            for q in 0..hw {
                assert!((out[base + q] - input[base + q] * scale).abs() < 1e-4, "kept {slot}/{q}");
                assert!((igrad[base + q] - seed[base + q] * scale).abs() < 1e-4, "kept grad {slot}/{q}");
            }
        }
    }
    let frac = dropped as f32 / (n * c) as f32;
    assert!((frac - p).abs() < 0.15, "dropped channel fraction {frac} should be near {p}");
    println!("resident dropout2d (train): whole-channel masking + grad routing, dropped {:.0}% of channels", frac * 100.0);
}

#[test]
fn cuda_dropout2d_eval_passthrough() {
    let (n, c, h, w) = (2usize, 3usize, 4usize, 4usize);
    let total = n * c * h * w;
    let input: Vec<f32> = (0..total).map(|i| (i % 13) as f32 * 0.1 - 0.5).collect();
    let gi = Node::new(input.clone(), vec![n, c, h, w]);
    gpu::to_cuda(&gi);
    let gout = dropout2d(&gi, 0.5, false);
    let out = gpu::to_host(&gout);
    for i in 0..total {
        assert!((out[i] - input[i]).abs() < 1e-6, "eval passthrough {i}");
    }
    println!("resident dropout2d (eval): passthrough");
}