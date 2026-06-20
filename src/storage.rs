use crate::device::Device;

/// The raw number storage behind a [`Tensor`].
///
/// Just a flat `Vec<f32>` plus the device it lives on (CPU for now). This is the
/// lowest level — the actual bytes. Shape and gradients live one step up in
/// [`Tensor`].
///
/// [`Tensor`]: crate::tensor::Tensor
#[derive(Clone, Debug)]
pub struct Storage {
    pub data: Vec<f32>,
    pub device: Device,
}

impl Storage {
    /// Wraps a flat vector of numbers as CPU storage.
    pub fn new(data: Vec<f32>) -> Self {
        Storage {
            data,
            device: Device::CPU,
        }
    }
}