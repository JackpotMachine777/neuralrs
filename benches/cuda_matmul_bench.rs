#[cfg(feature = "cuda")]
use criterion::{Criterion, Throughput};
#[cfg(feature = "cuda")]
use neuralrs::{cuda, ops::matmul::matmul as cpu_matmul, tensor::Tensor};

#[cfg(feature = "cuda")]
fn cuda_matmul_benchmark(c: &mut Criterion) {
    let n = 1024usize;
    let flops = 2u64 * n as u64 * n as u64 * n as u64;

    let a: Vec<f32> = (0..n * n).map(|i| (i % 1000) as f32 / 500.0 - 1.0).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 997) as f32 / 498.0 - 1.0).collect();
    let a_t = Tensor::new(a.clone(), vec![n, n]);
    let b_t = Tensor::new(b.clone(), vec![n, n]);

    let mut group = c.benchmark_group("matmul 1024x1024");
    group.sample_size(10);
    group.throughput(Throughput::Elements(flops));

    group.bench_function("cpu_simd", |bn| bn.iter(|| cpu_matmul(&a_t, &b_t)));
    group.bench_function("gpu_naive", |bn| bn.iter(|| cuda::matmul_naive(&a, &b, n, n, n)));
    group.bench_function("gpu_tiled", |bn| bn.iter(|| cuda::matmul(&a, &b, n, n, n)));

    group.finish();
}

#[cfg(feature = "cuda")]
fn main() {
    let mut c = Criterion::default().configure_from_args();
    cuda_matmul_benchmark(&mut c);
    c.final_summary();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("run with: cargo bench --bench cuda_matmul_bench --features cuda");
}