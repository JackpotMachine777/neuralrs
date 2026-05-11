use crate::tensor::Tensor;

pub fn transpose(t: &Tensor) -> Tensor{
    let rows = t.shape[0];
    let cols = t.shape[1];
    
    let mut out = vec![0.0; t.data.len()];

    for r in 0..rows{
        for c in 0..cols{
            out[c * rows + r] = t.data[r * cols + c];
        }
    }

    Tensor {
        data: out,
        grad: vec![0.0; t.data.len()],
        shape: vec![cols, rows],
    }
}