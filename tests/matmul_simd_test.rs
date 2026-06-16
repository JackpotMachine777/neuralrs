use rstorch::tensor::Tensor;
use rstorch::ops::matmul::matmul;
use rstorch::ops::simd::matmul_simd;

#[test]
fn matmul_simd_matches_naive() {
    let m = 5;
    let k = 11;
    let n = 7;

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32 - 4.0) * 0.3).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 6) as f32 - 2.0) * 0.25).collect();

    let a = Tensor::new(a_data, vec![m, k]);
    let b = Tensor::new(b_data, vec![k, n]);

    let res_naive = matmul(&a, &b);
    let res_simd = matmul_simd(&a, &b);

    println!("naive: {:?}", res_naive.storage.data);
    println!("simd:  {:?}", res_simd.storage.data);

    assert_eq!(res_simd.shape, res_naive.shape);
    for i in 0..res_naive.storage.data.len() {
        let diff = (res_simd.storage.data[i] - res_naive.storage.data[i]).abs();
        assert!(diff < 1e-3, "mismatch at {}: simd={} naive={}", i, res_simd.storage.data[i], res_naive.storage.data[i]);
    }

    println!("matmul_simd matches naive ok");
}