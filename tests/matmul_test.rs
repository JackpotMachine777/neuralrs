use rstorch::tensor::Tensor;
use rstorch::ops::matmul::matmul;

#[test]
fn matmul_basic() {
    let a = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
    );

    let c = matmul(&a, &b);

    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
}