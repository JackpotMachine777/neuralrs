use neuralrs::tensor::Tensor;
use neuralrs::nn::linear::Linear;
use neuralrs::serialize::{save_linear, load_linear, save_tensors, load_tensors, save, load};

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

#[test]
fn save_load_safetensors_roundtrip() {
    let a = vec![1.5f32, -2.25, 0.0, 3.75, 0.125, -1.0];
    let b = vec![0.5f32; 12];
    let path = "/tmp/test_roundtrip.safetensors";

    save(&[("layer.weight", &[2, 3][..], a.as_slice()), ("layer.bias", &[12][..], b.as_slice())], path);
    let loaded = load(path);

    assert_eq!(loaded.len(), 2);
    // safetensors files are name-sorted: bias comes back before weight
    assert_eq!(loaded[0].0, "layer.bias");
    assert_eq!(loaded[0].1, vec![12]);
    assert_eq!(loaded[0].2, b);
    assert_eq!(loaded[1].0, "layer.weight");
    assert_eq!(loaded[1].1, vec![2, 3]);
    assert_eq!(loaded[1].2, a);
}

#[test]
fn save_load_text_via_unified_api() {
    let a = vec![1.0f32, 2.0, 3.0];
    let path = "/tmp/test_unified.txt";
    save(&[("anything", &[3][..], a.as_slice())], path);
    let loaded = load(path);
    assert_eq!(loaded[0].0, "tensor_0");   // the text format drops names
    assert_eq!(loaded[0].1, vec![3]);
    assert_eq!(loaded[0].2, a);
}