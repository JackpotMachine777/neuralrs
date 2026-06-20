use neuralrs::data::mnist::{read_images, read_labels, read_labels_raw};

#[test]
fn mnist_loads_correctly() {
    let (images, count, rows, cols) = read_images("data/mnist/t10k-images-idx3-ubyte");

    println!("count: {count}, rows: {rows}, cols: {cols}");
    assert_eq!(count, 10000);
    assert_eq!(rows, 28);
    assert_eq!(cols, 28);
    assert_eq!(images.len(), 10000);
    assert_eq!(images[0].len(), 784);

    let max_pixel = images[0].iter().cloned().fold(0.0f32, f32::max);
    let min_pixel = images[0].iter().cloned().fold(1.0f32, f32::min);
    println!("first image pixel range: {min_pixel} to {max_pixel}");
    assert!(max_pixel <= 1.0);
    assert!(min_pixel >= 0.0);
    assert!(max_pixel > 0.0);

    let labels = read_labels("data/mnist/t10k-labels-idx1-ubyte");
    assert_eq!(labels.len(), 10000);
    assert_eq!(labels[0].len(), 10);
    let sum: f32 = labels[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    let raw = read_labels_raw("data/mnist/t10k-labels-idx1-ubyte");
    assert_eq!(raw.len(), 10000);
    println!("first 10 labels: {:?}", &raw[0..10]);
    assert!(raw.iter().all(|&d| d < 10));

    let first_digit = raw[0];
    assert_eq!(labels[0][first_digit], 1.0);

    println!("MNIST parser ok");
}