use crate::tensor::Tensor;

pub fn sum(t: &Tensor) -> f32 {
    let mut out = 0.0;

    for i in 0..t.data.len(){
        out += t.data[i];
    }

    out
}

pub fn mean(t: &Tensor) -> f32 {
    let mut out = 0.0;

    for i in 0..t.data.len(){
        out += t.data[i];
    }

    out / t.data.len() as f32
}