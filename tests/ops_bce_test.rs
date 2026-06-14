use rstorch::tensor::Tensor;
use rstorch::ops::elementwise::bce::{bce, bce_grad};

#[test]
fn bce_test(){
    let pred = Tensor::new(vec![0.8, 0.2], vec![2]);
    let target = Tensor::new(vec![1.0, 0.0], vec![2]);

    let loss = bce(&pred, &target);
    let grad = bce_grad(&pred, &target);

    println!("BCE loss: {}", loss);
    println!("BCE grad: {:?}", grad.data);

    assert!(loss > 0.0);
    assert_eq!(grad.shape, vec![2]);
}