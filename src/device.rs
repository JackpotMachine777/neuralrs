/// Where a tensor's data lives. `CUDA(i)` refers to the CUDA device at index
/// `i`; the resident GPU backend currently always uses device 0.
#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),
}