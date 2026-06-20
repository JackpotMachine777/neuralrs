use neuralrs::tensor::Tensor;
use neuralrs::ops::matmul::matmul;

#[test]
fn matmul_correct_small() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c = matmul(&a, &b);

    println!("result: {:?}", c.storage.data);
    assert_eq!(c.shape, vec![2, 2]);
    assert!((c.storage.data[0] - 58.0).abs() < 1e-4, "C[0,0]");
    assert!((c.storage.data[1] - 64.0).abs() < 1e-4, "C[0,1]");
    assert!((c.storage.data[2] - 139.0).abs() < 1e-4, "C[1,0]");
    assert!((c.storage.data[3] - 154.0).abs() < 1e-4, "C[1,1]");
    println!("matmul correct (small) ok");
}

#[test]
fn matmul_correct_square() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = matmul(&a, &b);

    assert_eq!(c.shape, vec![2, 2]);
    assert!((c.storage.data[0] - 19.0).abs() < 1e-4);
    assert!((c.storage.data[1] - 22.0).abs() < 1e-4);
    assert!((c.storage.data[2] - 43.0).abs() < 1e-4);
    assert!((c.storage.data[3] - 50.0).abs() < 1e-4);
    println!("matmul correct (square) ok");
}

#[test]
fn matmul_non_square() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::new(vec![1.0,2.0,3.0,4.0, 5.0,6.0,7.0,8.0, 9.0,10.0,11.0,12.0], vec![3, 4]);
    let c = matmul(&a, &b);

    println!("non-square result: {:?}", c.storage.data);
    assert_eq!(c.shape, vec![1, 4]);
    assert!((c.storage.data[0] - 38.0).abs() < 1e-4);
    assert!((c.storage.data[1] - 44.0).abs() < 1e-4);
    assert!((c.storage.data[2] - 50.0).abs() < 1e-4);
    assert!((c.storage.data[3] - 56.0).abs() < 1e-4);
    println!("matmul non-square ok");
}