#[derive(Clone, Debug, PartialEq)]
/// Where a tensor's data lives. CPU only for now (GPU support would be a future
/// addition).
pub enum Device {
    CPU,
    CUDA(usize),
}