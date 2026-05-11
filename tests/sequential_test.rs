use rstorch::tensor::Tensor;
use rstorch::nn::{module::Module, sequential::Sequential};
use rstorch::nn::linear::Linear;
use rstorch::nn::activations::ReLU;

#[test]
fn sequential_forward_test() {
    let model = Sequential {
        list: vec![
            Box::new(Linear {
                weights: Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
            }),
            Box::new(ReLU {}),
            Box::new(Linear {
                weights: Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
            }),
        ],
    };

    let input = Tensor::new(vec![1.0, -2.0], vec![1, 2]);

    let output = model.forward(&input);

    println!("{:?}", output);
}