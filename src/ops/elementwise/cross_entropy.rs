use crate::tensor::Tensor;

pub fn cross_entropy(pred: &Tensor, target: &Tensor) -> f32 {
    if pred.shape != target.shape{
        panic!("Shapes are different");
    }

    let mut res = 0.0;

    for i in 0..pred.data.len() {
        res += target.data[i] * pred.data[i].ln()
    }

    -res / pred.data.len() as f32
}

pub fn cross_entropy_grad(pred: &Tensor, target: &Tensor) -> Tensor {
    let mut res = Vec::with_capacity(pred.data.len());

    for i in 0..pred.data.len() {
        res.push(-target.data[i] / pred.data[i]);
    }

    Tensor {
        data: res,
        grad: vec![0.0; pred.data.len()],
        shape: pred.shape.clone(),
    }
}