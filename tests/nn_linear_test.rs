use rstorch::tensor::Tensor;
use rstorch::nn::linear::Linear;
use rstorch::nn::module::Module;
use rstorch::autograd::node::Node;

#[test]
fn linear_forward_backward_test() {
    let mut layer = Linear {
        weights: Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        bias: Tensor::new(vec![0.0, 0.0], vec![2]),
        weights_node: None,
        bias_node: None,
    };

    let input = Node::new(vec![3.0, -2.0], vec![1, 2]);
    let output = layer.forward(input);

    println!("output: {:?}", output.borrow().data);
    assert_eq!(output.borrow().shape, vec![1, 2]);

    let n = output.borrow().data.len();
    output.borrow_mut().grad = vec![1.0; n];
    output.borrow_mut().backward();

    layer.sync_grads();

    assert!(layer.weights.grad.iter().any(|&x| x != 0.0));
    assert!(layer.bias.grad.iter().any(|&x| x != 0.0));
}