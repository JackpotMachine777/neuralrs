use crate::tensor::Tensor;

impl Tensor{
    // Creating a new tensor //
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self{
        let count: usize = shape.iter().product();

        if count != data.len(){
            panic!("Shape does not match data size");
        }

        let len = data.len();
        Tensor { data, grad: vec![0.0; len], shape }
    }

    // Adding tensors //
    pub fn add(&self, other: &Tensor) -> Tensor{
        if self.shape == other.shape{
            let mut result = Vec::with_capacity(self.data.len());

            for i in 0..self.data.len(){
                result.push(self.data[i] + other.data[i]);
            }

            return Tensor{
                data: result,
                grad: vec![0.0; self.data.len()],
                shape: self.shape.clone(),
            };
        }

        if other.shape.len() == 1 && self.shape.len() == 2{
            let features = other.shape[0];
            let batch = self.shape[0];

            if self.shape[1] != features{
                panic!("Broadcast failed: feature size mismatch");
            }

            let mut res = Vec::with_capacity(self.data.len());

            for b in 0..batch{
                for f in 0..features{
                    let i = b * features + f;
                    res.push(self.data[i] + other.data[f]);
                }
            }

            return Tensor{
                data: res,
                grad: vec![0.0; self.data.len()],
                shape: self.shape.clone(),
            }
        }

        panic!("Shapes are not compatible for add");
    }

    // Multiplying tensors //
    pub fn mul(&self, other: &Tensor) -> Tensor{
        if self.shape != other.shape{
            panic!("Shapes are different");
        }

        let mut result = Vec::with_capacity(self.data.len());

        for i in 0..self.data.len(){
            result.push(self.data[i] * other.data[i]);
        }

        Tensor {
            data: result,
            grad: vec![0.0; self.data.len()],
            shape: self.shape.clone(),
        }
    }
}