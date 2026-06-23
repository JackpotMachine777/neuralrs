#[cfg(feature = "cuda")]

use neuralrs::cuda;
use neuralrs::tensor::Tensor;

#[test]
fn cuda_add_matches_cpu() {
    let n: usize = 10_000;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.02).collect();

    let cpu = Tensor::new(a.clone(), vec![n]).add(&Tensor::new(b.clone(), vec![n]));
    let gpu = cuda::add(&a, &b);

    assert_eq!(gpu.len(), cpu.storage.data.len());
    for i in 0..n {
        assert!(
            (gpu[i] - cpu.storage.data[i]).abs() < 1e-5,
            "mismatch at {i}: gpu {} cpu {}",
            gpu[i],
            cpu.storage.data[i]
        );
    }
    println!("cuda add matches cpu over {n} elements");
}