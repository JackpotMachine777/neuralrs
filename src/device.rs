#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),
}