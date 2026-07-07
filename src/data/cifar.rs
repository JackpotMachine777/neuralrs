//! CIFAR-10 loader for the binary dataset format.
//!
//! Each `.bin` file is a flat sequence of 3073-byte records: one label byte
//! (0–9) followed by 3072 pixel bytes laid out as [channel, row, col], 1024
//! red, then 1024 green, then 1024 blue, which is already the [C, H, W] order
//! `conv2d` expects, so no reshaping is needed. Pixels are normalized to
//! [0, 1]. Mirrors the shape of the MNIST loader.

use std::fs::File;
use std::io::{BufReader, Read};

const RECORD: usize = 3073; // 1 label byte + 3 * 32 * 32 pixel bytes
const IMG: usize = 3072;

/// Reads CIFAR-10 images from one or more binary batch files into flat `f32`
/// vectors of length 3072 (`[3, 32, 32]`, normalized), concatenated in the
/// order the paths are given.
pub fn read_images(paths: &[&str]) -> Vec<Vec<f32>> {
    let mut images = Vec::new();
    for path in paths {
        let file = File::open(path).unwrap_or_else(|_| panic!("cannot open CIFAR file {path}"));
        let mut reader = BufReader::new(file);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).expect("read failed");
        assert_eq!(buf.len() % RECORD, 0, "CIFAR file {path} is not a whole number of records");

        for rec in buf.chunks_exact(RECORD) {
            let img: Vec<f32> = rec[1..1 + IMG].iter().map(|&b| b as f32 / 255.0).collect();
            images.push(img);
        }
    }
    images
}

/// Reads CIFAR-10 labels as one-hot vectors (length 10), ready as targets.
pub fn read_labels(paths: &[&str]) -> Vec<Vec<f32>> {
    let mut labels = Vec::new();
    for path in paths {
        let file = File::open(path).unwrap_or_else(|_| panic!("cannot open CIFAR file {path}"));
        let mut reader = BufReader::new(file);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).expect("read failed");

        for rec in buf.chunks_exact(RECORD) {
            let mut one_hot = vec![0.0; 10];
            one_hot[rec[0] as usize] = 1.0;
            labels.push(one_hot);
        }
    }
    labels
}

/// Reads CIFAR-10 labels as plain class indices (0–9), for accuracy checks.
pub fn read_labels_raw(paths: &[&str]) -> Vec<usize> {
    let mut labels = Vec::new();
    for path in paths {
        let file = File::open(path).unwrap_or_else(|_| panic!("cannot open CIFAR file {path}"));
        let mut reader = BufReader::new(file);
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).expect("read failed");

        for rec in buf.chunks_exact(RECORD) {
            labels.push(rec[0] as usize);
        }
    }
    labels
}