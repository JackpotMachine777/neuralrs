use std::fs::File;
use std::io::{Read, BufReader};

/// Reads MNIST images from the IDX file format into a list of flat `f32` vectors
/// (pixel values normalized), plus the image dimensions.
pub fn read_images(path: &str) -> (Vec<Vec<f32>>, usize, usize, usize) {
    let file = File::open(path).expect("cannot open MNIST image file");
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read failed");

    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;

    let img_size = rows * cols;
    let mut images = Vec::with_capacity(count);

    for i in 0..count {
        let start = 16 + i * img_size;
        let img: Vec<f32> = buf[start..start + img_size]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(img);
    }

    (images, count, rows, cols)
}

/// Reads MNIST labels as one-hot vectors (length 10), ready to use as targets.
pub fn read_labels(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("cannot open MNIST label file");
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read failed");

    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    let mut labels = Vec::with_capacity(count);
    for i in 0..count {
        let digit = buf[8 + i] as usize;
        let mut one_hot = vec![0.0; 10];
        one_hot[digit] = 1.0;
        labels.push(one_hot);
    }

    labels
}

/// Reads MNIST labels as plain digit indices (0–9), handy for accuracy checks.
pub fn read_labels_raw(path: &str) -> Vec<usize> {
    let file = File::open(path).expect("cannot open MNIST label file");
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read failed");

    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    (0..count).map(|i| buf[8 + i] as usize).collect()
}