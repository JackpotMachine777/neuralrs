#[derive(Clone, Debug)]
pub struct Tensor{
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    // dtype: DType,
}