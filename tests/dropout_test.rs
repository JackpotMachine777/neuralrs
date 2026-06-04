use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::dropout::Dropout;

#[test]
fn dropout_test(){
    let mut layer = Dropout {
        probability: 0.5,
        mask: vec![],
        training: true,
    };

    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let output = layer.forward(&input);

    println!("input:  {:?}", input.data);
    println!("output: {:?}", output.data);

    assert_eq!(output.shape, input.shape);
}