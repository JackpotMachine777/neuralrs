//! Plain matrix multiplication (no autograd).
//!
//! This is the fast, gradient-free matmul used under the hood. It has three
//! implementations picked at compile time: an optional BLAS backend
//! (`--features blas`), a hand-written AVX2 SIMD version on x86-64 (the
//! default), and a parallel fallback everywhere else. The autograd-aware matmul
//! that builds graph nodes lives in `autograd::graph::matmul`.

use crate::tensor::Tensor;

#[cfg(feature = "blas")]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    use crate::storage::Storage;
    use crate::dtype::DType;

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    if k != b.shape[0] {
        panic!("Shapes are not matching");
    }

    let a_data = &a.storage.data;
    let b_data = &b.storage.data;
    let mut c = vec![0.0f32; m * n];

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a_data.as_ptr(), k as isize, 1,
            b_data.as_ptr(), n as isize, 1,
            0.0,
            c.as_mut_ptr(), n as isize, 1,
        );
    }

    Tensor {
        storage: Storage::new(c),
        grad: vec![0.0; m * n],
        shape: vec![m, n],
        dtype: DType::Float32,
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    use crate::ops::simd::matmul_simd;

    if a.shape[1] != b.shape[0] {
        panic!("Shapes are not matching");
    }

    matmul_simd(a, b)
}

#[cfg(all(not(target_arch = "x86_64"), not(feature = "blas")))]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    use crate::storage::Storage;
    use crate::dtype::DType;
    use rayon::prelude::*;

    let m = a.shape[0];
    let n = b.shape[1];
    let k = a.shape[1];

    if k != b.shape[0] {
        panic!("Shapes are not matching");
    }

    let mut res = vec![0.0; m * n];
    let a_data = &a.storage.data;
    let b_data = &b.storage.data;

    res.par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for t in 0..k {
                    let x = a_data[i * k + t];
                    let y = b_data[t * n + j];
                    sum += x * y;
                }
                row[j] = sum;
            }
        });

    Tensor {
        storage: Storage::new(res),
        grad: vec![0.0; m * n],
        shape: vec![m, n],
        dtype: DType::Float32,
    }
}