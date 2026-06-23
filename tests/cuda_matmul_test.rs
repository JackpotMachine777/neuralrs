#![cfg(feature = "cuda")]

use neuralrs::cuda;
use neuralrs::ops::matmul::matmul as cpu_matmul;
use neuralrs::tensor::Tensor;

fn assert_close(gpu: &[f32], cpu: &[f32], eps: f32, label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for i in 0..gpu.len() {
        assert!(
            (gpu[i] - cpu[i]).abs() < eps,
            "{label} mismatch at {i}: gpu {} cpu {}",
            gpu[i],
            cpu[i]
        );
    }
}

#[test]
fn cuda_matmul_small_exact() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let (m, k, n) = (2, 3, 2);
    let cpu = cpu_matmul(&a, &b);

    let tiled = cuda::matmul(&a.storage.data, &b.storage.data, m, k, n);
    let naive = cuda::matmul_naive(&a.storage.data, &b.storage.data, m, k, n);

    assert_close(&tiled, &cpu.storage.data, 1e-4, "tiled");
    assert_close(&naive, &cpu.storage.data, 1e-4, "naive");
    println!("small matmul gpu == cpu -> {tiled:?}");
}

#[test]
fn cuda_matmul_irregular_shapes() {
    let (m, k, n) = (50usize, 70usize, 30usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
    let at = Tensor::new(a.clone(), vec![m, k]);
    let bt = Tensor::new(b.clone(), vec![k, n]);
    let cpu = cpu_matmul(&at, &bt);

    let tiled = cuda::matmul(&a, &b, m, k, n);
    let naive = cuda::matmul_naive(&a, &b, m, k, n);

    assert_close(&tiled, &cpu.storage.data, 1e-2, "tiled");
    assert_close(&naive, &cpu.storage.data, 1e-2, "naive");
    println!("irregular {m}x{k} * {k}x{n} gpu == cpu");
}