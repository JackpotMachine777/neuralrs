use rstorch::tensor::Tensor;
use rstorch::ops::matmul::matmul;
use std::time::Instant;

#[test]
fn bench_matmul() {
    let n = 512;
    let a_data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 * 0.2).collect();

    let a = Tensor::new(a_data, vec![n, n]);
    let b = Tensor::new(b_data, vec![n, n]);

    let start = Instant::now();
    let c = matmul(&a, &b);
    let elapsed = start.elapsed();

    println!("checksum: {}", c.storage.data[0] + c.storage.data[n * n - 1]);
    println!("matmul {}x{} took: {:?}", n, n, elapsed);
}