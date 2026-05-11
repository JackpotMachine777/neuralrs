use rstorch::tensor::Tensor;
use rstorch::nn::linear::Linear;
use rstorch::nn::module::Module;

#[test]
fn linear_forward_backward_test() {
    let mut layer = Linear {
        weights: Tensor::new(vec![
            1.0, 0.0,
            0.0, 1.0,
        ], vec![2, 2]),
        bias: Tensor::new(vec![0.0, 0.0], vec![2]),
        last_input: None,
    };

    let input = Tensor::new(vec![3.0, -2.0], vec![1, 2]);

    let output = layer.forward(&input);

    assert_eq!(output.shape, vec![1, 2]);

    let grad_output = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let grad_input = layer.backward(&grad_output);

    assert_eq!(grad_input.shape, vec![1, 2]);

    assert!(layer.weights.grad.iter().any(|&x| x != 0.0));
    assert!(layer.bias.grad.iter().any(|&x| x != 0.0));
}