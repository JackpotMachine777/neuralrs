use crate::tensor::Tensor;

pub fn add_vec(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>{
    let mut out = vec![0.0; a.len()];

    for i in 0..a.len(){
        out[i] = a[i] + b[i];
    }

    out
}

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