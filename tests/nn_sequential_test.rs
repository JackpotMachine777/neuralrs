use neuralrs::tensor::Tensor;
use neuralrs::nn::{module::Module, sequential::Sequential};
use neuralrs::nn::linear::Linear;
use neuralrs::nn::activations::relu::ReLU;
use neuralrs::autograd::node::Node;

#[test]
fn sequential_forward_test() {
    let mut model = Sequential {
        list: vec![
            Box::new(Linear {
                weights: Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                weights_node: None,
                bias_node: None,
            }),
            Box::new(ReLU {}),
            Box::new(Linear {
                weights: Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                weights_node: None,
                bias_node: None,
            }),
        ],
    };

    let input = Node::new(vec![1.0, -2.0], vec![1, 2]);
    let output = model.forward(input);

    println!("{:?}", output.borrow().data);
}