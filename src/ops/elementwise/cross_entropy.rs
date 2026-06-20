use crate::tensor::Tensor;
use crate::storage::Storage;
use crate::dtype::DType;

/// Cross-entropy loss between predicted probabilities and one-hot targets.
pub fn cross_entropy(pred: &Tensor, target: &Tensor) -> f32 {
    if pred.shape != target.shape{
        panic!("Shapes are different");
    }

    let mut res = 0.0;

    for i in 0..pred.storage.data.len() {
        res += target.storage.data[i] * pred.storage.data[i].ln()
    }

    -res / pred.storage.data.len() as f32
}

pub fn cross_entropy_grad(pred: &Tensor, target: &Tensor) -> Tensor {
    let mut res = Vec::with_capacity(pred.storage.data.len());

    for i in 0..pred.storage.data.len() {
        res.push(-target.storage.data[i] / pred.storage.data[i]);
    }

    Tensor {
        storage: Storage::new(res),
        grad: vec![0.0; pred.storage.data.len()],
        shape: pred.shape.clone(),
        dtype: DType::Float32,
    }
}