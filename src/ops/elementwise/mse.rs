use crate::tensor::Tensor;

pub fn mse(pred: &Tensor, target: &Tensor) -> f32{
    if pred.shape != target.shape{
        panic!("Shapes are different");
    }

    let mut res = 0.0;

    for i in 0..pred.data.len(){
        let diff = pred.data[i] - target.data[i];
        res += diff * diff;
    }

    res / pred.data.len() as f32
}

pub fn mse_grad(pred: &Tensor, target: &Tensor) -> Tensor{
    let mut res = Vec::with_capacity(pred.data.len());

    for i in 0..pred.data.len(){
        res.push(2.0 * (pred.data[i] - target.data[i]) / pred.data.len() as f32);
    }

    Tensor {
        data: res,
        grad: vec![0.0; pred.data.len()],
        shape: pred.shape.clone(),
    }
}