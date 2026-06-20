use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::linear::Linear;
use neuralrs::autograd::node::Node;

#[test]
fn linear_batch_test() {
    let mut layer = Linear {
        weights: Tensor::new(vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ], vec![3, 2]),
        bias: Tensor::new(vec![10.0, 20.0], vec![2]),
        weights_node: None,
        bias_node: None,
    };

    let input = Node::new(vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ], vec![2, 3]);

    let output = layer.forward(input.clone());

    println!("output: {:?}", output.borrow().data);
    println!("shape:  {:?}", output.borrow().shape);

    assert_eq!(output.borrow().shape, vec![2, 2]);
    assert_eq!(output.borrow().data, vec![14.0, 25.0, 20.0, 31.0]);

    output.borrow_mut().grad = vec![1.0, 1.0, 1.0, 1.0];
    output.borrow_mut().backward();

    layer.sync_grads();
    println!("bias grad: {:?}", layer.bias.grad);
    assert_eq!(layer.bias.grad, vec![2.0, 2.0]);

    println!("linear batch ok");
}