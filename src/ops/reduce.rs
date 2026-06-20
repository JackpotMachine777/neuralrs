use crate::tensor::Tensor;

/// Sums all elements of a tensor into a single value.
pub fn sum(t: &Tensor) -> f32 {
    let mut out = 0.0;

    for i in 0..t.storage.data.len(){
        out += t.storage.data[i];
    }

    out
}

/// Averages all elements of a tensor.
pub fn mean(t: &Tensor) -> f32 {
    let mut out = 0.0;

    for i in 0..t.storage.data.len(){
        out += t.storage.data[i];
    }

    out / t.storage.data.len() as f32
}