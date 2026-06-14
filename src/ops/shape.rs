use crate::tensor::Tensor;
use crate::storage::Storage;
use crate::dtype::DType;

pub fn transpose(t: &Tensor) -> Tensor{
    let rows = t.shape[0];
    let cols = t.shape[1];
    
    let mut out = vec![0.0; t.storage.data.len()];

    for r in 0..rows{
        for c in 0..cols{
            out[c * rows + r] = t.storage.data[r * cols + c];
        }
    }

    Tensor {
        storage: Storage::new(out),
        grad: vec![0.0; t.storage.data.len()],
        shape: vec![cols, rows],
        dtype: DType::Float32,
    }
}