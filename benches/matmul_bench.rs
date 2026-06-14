use criterion::{criterion_group, criterion_main, Criterion};
use rstorch::tensor::Tensor;
use rstorch::ops::matmul::matmul;

fn matmul_benchmark(c: &mut Criterion) {
    let n = 512;
    let a_data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 * 0.2).collect();
    let a = Tensor::new(a_data, vec![n, n]);
    let b = Tensor::new(b_data, vec![n, n]);

    c.bench_function("matmul 512x512", |bench| {
        bench.iter(|| matmul(&a, &b))
    });
}

criterion_group!(benches, matmul_benchmark);
criterion_main!(benches);