use crate::tensor::Tensor;

#[cfg(target_arch = "x86_64")]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor{
    use crate::ops::simd::matmul_simd;
    
    if a.shape[1] != b.shape[0] {
        panic!("Shapes are not matching");
    }

    matmul_simd(a, b)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    use crate::storage::Storage;
    use crate::dtype::DType;
    use rayon::prelude::*;

    let m = a.shape[0];
    let n = b.shape[1];
    let k = a.shape[1];

    if k != b.shape[0]{
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