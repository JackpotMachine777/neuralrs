use neuralrs::tensor::Tensor;
use neuralrs::ops::matmul::matmul;

#[test]
fn tensor_creation() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.storage.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_add() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![4]);

    let c = a.add(&b);

    assert_eq!(c.storage.data, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn tensor_mul() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2]);
    let b = Tensor::new(vec![3.0, 4.0], vec![2]);

    let c = a.mul(&b);

    assert_eq!(c.storage.data, vec![3.0, 8.0]);
}

#[test]
fn tensor_matmul() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = matmul(&a, &b);

    assert_eq!(c.storage.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
#[should_panic]
fn tensor_shape_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![1.0, 2.0], vec![2]);

    let _ = a.add(&b);
}