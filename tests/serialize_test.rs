use neuralrs::tensor::Tensor;
use neuralrs::nn::linear::Linear;
use neuralrs::serialize::{save_linear, load_linear, save_tensors, load_tensors};

#[test]
fn save_load_linear_test() {
    let original = Linear {
        weights: Tensor::new(vec![1.5, -2.3, 0.7, 4.1, -0.9, 3.3], vec![2, 3]),
        bias: Tensor::new(vec![0.5, -1.2], vec![2]),
        weights_node: None,
        bias_node: None,
    };

    let path = "/tmp/test_linear.txt";

    save_linear(&original, path);

    let loaded = load_linear(path);

    assert_eq!(loaded.weights.shape, vec![2, 3]);
    assert_eq!(loaded.bias.shape, vec![2]);
    assert_eq!(loaded.weights.storage.data, original.weights.storage.data);
    assert_eq!(loaded.bias.storage.data, original.bias.storage.data);

    println!("original weights: {:?}", original.weights.storage.data);
    println!("loaded weights:   {:?}", loaded.weights.storage.data);
    println!("save/load ok");
}

#[test]
fn save_load_tensors_roundtrip() {
    let a = vec![1.5f32, -2.25, 0.0, 3.75];
    let b = vec![0.125f32; 7];
    let path = "/tmp/test_tensors.txt";

    save_tensors(&[a.as_slice(), b.as_slice()], path);
    let loaded = load_tensors(path);

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0], a);
    assert_eq!(loaded[1], b);
}