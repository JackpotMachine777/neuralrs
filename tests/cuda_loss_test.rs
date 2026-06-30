#![cfg(feature = "cuda")]

use neuralrs::autograd::node::Node;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::loss::{bce as gpu_bce, bce_backward, mse as gpu_mse, mse_backward};
use neuralrs::ops::elementwise::bce::{bce as cpu_bce, bce_grad as cpu_bce_grad};
use neuralrs::ops::elementwise::mse::{mse as cpu_mse, mse_grad as cpu_mse_grad};
use neuralrs::tensor::Tensor;

#[test]
fn cuda_mse() {
    let n = 256;
    let pred: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let target: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.5).collect();

    let pt = Tensor::new(pred.clone(), vec![n]);
    let tt = Tensor::new(target.clone(), vec![n]);
    let c_loss = cpu_mse(&pt, &tt);
    let c_grad = cpu_mse_grad(&pt, &tt).storage.data;

    let pn = Node::new(pred, vec![n]);
    let tn = Node::new(target, vec![n]);
    gpu::to_cuda(&pn);
    gpu::to_cuda(&tn);
    let g_loss = gpu_mse(&pn, &tn);
    mse_backward(&pn, &tn);
    let g_grad = gpu::read_grad(&pn);

    assert!((g_loss - c_loss).abs() < 1e-3, "mse loss: gpu {g_loss} cpu {c_loss}");
    for i in 0..n {
        assert!((g_grad[i] - c_grad[i]).abs() < 1e-4, "mse grad {i}: gpu {} cpu {}", g_grad[i], c_grad[i]);
    }
    println!("mse: loss gpu {:.6} cpu {:.6}, grad matches", g_loss, c_loss);
}

#[test]
fn cuda_bce() {
    let n = 256;
    let pred: Vec<f32> = (0..n).map(|i| 0.1 + (i % 9) as f32 * 0.1).collect();
    let target: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();

    let pt = Tensor::new(pred.clone(), vec![n]);
    let tt = Tensor::new(target.clone(), vec![n]);
    let c_loss = cpu_bce(&pt, &tt);
    let c_grad = cpu_bce_grad(&pt, &tt).storage.data;

    let pn = Node::new(pred, vec![n]);
    let tn = Node::new(target, vec![n]);
    gpu::to_cuda(&pn);
    gpu::to_cuda(&tn);
    let g_loss = gpu_bce(&pn, &tn);
    bce_backward(&pn, &tn);
    let g_grad = gpu::read_grad(&pn);

    assert!((g_loss - c_loss).abs() < 1e-3, "bce loss: gpu {g_loss} cpu {c_loss}");
    for i in 0..n {
        assert!((g_grad[i] - c_grad[i]).abs() < 1e-3, "bce grad {i}: gpu {} cpu {}", g_grad[i], c_grad[i]);
    }
    println!("bce: loss gpu {:.6} cpu {:.6}, grad matches", g_loss, c_loss);
}