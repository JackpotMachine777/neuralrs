use crate::tensor::Tensor;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor{
        let m = a.shape[0];
        let n = b.shape[1];
        let k = a.shape[1];

        if k != b.shape[0]{
            panic!("Shapes are not matching");
        }

        let mut res = vec![0.0; m * n];

        for i in 0..m{
            for j in 0..n{
                let mut sum = 0.0;

                for t in 0..k{
                    let x = a.data[i * k + t];
                    let y = b.data[t * n + j];
                    sum += x * y;
                }

                res[i * n + j] = sum;
            }
        }

        Tensor {
            data: res,
            shape: vec![m, n],
        }
    }