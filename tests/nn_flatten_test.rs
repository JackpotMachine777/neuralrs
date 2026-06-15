use rstorch::nn::module::Module;
use rstorch::nn::flatten::Flatten;
use rstorch::autograd::node::Node;

#[test]
fn flatten_test() {
    let mut layer = Flatten {};

    let input = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let output = layer.forward(input.clone());

    assert_eq!(output.borrow().shape, vec![2, 4]);
    assert_eq!(output.borrow().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    output.borrow_mut().grad = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    output.borrow_mut().backward();
    assert_eq!(input.borrow().grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    println!("flatten ok");
}