// pub enum DType{
//     Float32,
//     Float64,
//     Int32,
//     Int64,
//     Bool,
// }

pub struct Tensor{
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    // dtype: DType,
}

impl Tensor{
    // Creating a new tensor //
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self{
        let count: usize = shape.iter().product();

        if count != data.len(){
            panic!("Shape does not match data size");
        }

        Tensor { data, shape }
    }

    pub fn add(&self, other: &Tensor) -> Tensor{
        if self.shape != other.shape{
            panic!("Shapes are different");
        }

        let mut result = Vec::with_capacity(self.data.len());

        for i in 0..self.data.len(){
            result.push(self.data[i] + other.data[i]);
        }

        Tensor {
            data: result,
            shape: self.shape.clone(),
        }
    }

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
            shape: self.shape.clone(),
        }
    }
}