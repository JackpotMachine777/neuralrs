use crate::tensor::Tensor;
use crate::storage::Storage;
use crate::dtype::DType;

impl Tensor {
    /// Builds a tensor from a flat data vector and a shape.
    ///
    /// The shape has to match the data: multiply the dimensions together and you
    /// should get the number of values, otherwise it panics. The gradient buffer
    /// starts out all zeros.
    ///
    /// # Panics
    /// If `shape` doesn't match how many elements are in `data`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let count: usize = shape.iter().product();

        if count != data.len() {
            panic!("Shape does not match data size");
        }

        let len = data.len();

        Tensor {
            storage: Storage::new(data),
            grad: vec![0.0; len],
            shape,
            dtype: DType::Float32,
        }
    }

    /// Adds two tensors element by element, with bias-style broadcasting.
    ///
    /// Same shape? Adds them straight across. The one special case: a 1-D tensor
    /// `[features]` can be added to a 2-D tensor `[batch, features]`, and it gets
    /// reused for every row in the batch — that's the classic "add a bias vector
    /// to a batch of rows" move.
    ///
    /// Plain data op, doesn't track gradients.
    ///
    /// # Panics
    /// If the shapes aren't equal and aren't a valid bias broadcast either.
    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let mut result = Vec::with_capacity(self.storage.data.len());

            for i in 0..self.storage.data.len() {
                result.push(self.storage.data[i] + other.storage.data[i]);
            }

            return Tensor {
                storage: Storage::new(result),
                grad: vec![0.0; self.storage.data.len()],
                shape: self.shape.clone(),
                dtype: DType::Float32,
            };
        }

        if other.shape.len() == 1 && self.shape.len() == 2 {
            let features = other.shape[0];
            let batch = self.shape[0];

            if self.shape[1] != features {
                panic!("Broadcast failed: feature size mismatch");
            }

            let mut res = Vec::with_capacity(self.storage.data.len());

            for b in 0..batch {
                for f in 0..features {
                    let i = b * features + f;
                    res.push(self.storage.data[i] + other.storage.data[f]);
                }
            }

            return Tensor {
                storage: Storage::new(res),
                grad: vec![0.0; self.storage.data.len()],
                shape: self.shape.clone(),
                dtype: DType::Float32,
            };
        }

        panic!("Shapes are not compatible for add");
    }

    /// Multiplies two same-shape tensors element by element.
    ///
    /// Plain data op, doesn't track gradients.
    ///
    /// # Panics
    /// If the two shapes don't match.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.shape != other.shape {
            panic!("Shapes are different");
        }

        let mut result = Vec::with_capacity(self.storage.data.len());

        for i in 0..self.storage.data.len() {
            result.push(self.storage.data[i] * other.storage.data[i]);
        }
        
        Tensor {
            storage: Storage::new(result),
            grad: vec![0.0; self.storage.data.len()],
            shape: self.shape.clone(),
            dtype: DType::Float32,
        }
    }
}