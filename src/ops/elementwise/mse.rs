use crate::tensor::Tensor;
use crate::storage::Storage;
use crate::dtype::DType;

pub fn mse(pred: &Tensor, target: &Tensor) -> f32{
    if pred.shape != target.shape{
        panic!("Shapes are different");
    }

    let mut res = 0.0;

    for i in 0..pred.storage.data.len(){
        let diff = pred.storage.data[i] - target.storage.data[i];
        res += diff * diff;
    }

    res / pred.storage.data.len() as f32
}

pub fn mse_grad(pred: &Tensor, target: &Tensor) -> Tensor{
    let mut res = Vec::with_capacity(pred.storage.data.len());

    for i in 0..pred.storage.data.len(){
        res.push(2.0 * (pred.storage.data[i] - target.storage.data[i]) / pred.storage.data.len() as f32);
    }

    Tensor {
        storage: Storage::new(res),
        grad: vec![0.0; pred.storage.data.len()],
        shape: pred.shape.clone(),
        dtype: DType::Float32,
    }
}