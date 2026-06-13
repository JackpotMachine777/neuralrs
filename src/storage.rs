use crate::device::Device;

pub struct Storage {
    pub data: Vec<f32>,
    pub device: Device,
}

impl Storage {
    pub fn new(data: Vec<f32>) -> Self {
        Storage {
            data,
            device: Device::CPU,
        }
    }
}