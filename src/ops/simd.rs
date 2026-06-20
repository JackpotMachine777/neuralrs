use crate::tensor::Tensor;
use crate::storage::Storage;
use crate::dtype::DType;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
/// Dot product of two slices using AVX2 SIMD where available, for speed.
pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += a[tail_start + i] * b[tail_start + i];
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
/// SIMD-accelerated matrix multiplication — the default matmul backend on
/// x86-64.
pub fn matmul_simd(a: &Tensor, b: &Tensor) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    assert_eq!(k, b.shape[0], "matmul shape mismatch");

    let a_data = &a.storage.data;
    let b_data = &b.storage.data;

    let mut bt = vec![0.0f32; n * k];
    for row in 0..k {
        for col in 0..n {
            bt[col * k + row] = b_data[row * n + col];
        }
    }

    let mut res = vec![0.0f32; m * n];

    res.par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, out_row)| {
            let a_row = &a_data[i * k..i * k + k];

            for j in 0..n {
                let bt_row = &bt[j * k..j * k + k];
                out_row[j] = dot_simd(a_row, bt_row);
            }
        });

    Tensor {
        storage: Storage::new(res),
        grad: vec![0.0; m * n],
        shape: vec![m, n],
        dtype: DType::Float32,
    }
}