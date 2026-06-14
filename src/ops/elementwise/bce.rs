use crate::tensor::Tensor;
use crate::storage::Storage;
use crate::dtype::DType;

pub fn bce(pred: &Tensor, target: &Tensor) -> f32{
    if pred.shape != target.shape{
        panic!("Shapes are different");
    }

    let mut res = 0.0;

    for i in 0..pred.storage.data.len(){
        res += target.storage.data[i] * pred.storage.data[i].ln() + (1.0 - target.storage.data[i]) * (1.0 - pred.storage.data[i]).ln()
    }

    -res / pred.storage.data.len() as f32
}

pub fn bce_grad(pred: &Tensor, target: &Tensor) -> Tensor{
    let mut res = Vec::with_capacity(pred.storage.data.len());

    for i in 0..pred.storage.data.len(){
        res.push((pred.storage.data[i] - target.storage.data[i]) / (pred.storage.data[i] * (1.0 - pred.storage.data[i])));
    }

    Tensor {
        storage: Storage::new(res),
        grad: vec![0.0; pred.storage.data.len()],
        shape: pred.shape.clone(),
        dtype: DType::Float32,
    }
}