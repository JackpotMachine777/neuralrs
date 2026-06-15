use rstorch::nn::module::Module;
use rstorch::nn::maxpool::MaxPool2d;
use rstorch::autograd::node::Node;

#[test]
fn maxpool_test() {
    let mut layer = MaxPool2d {
        kernel: 2,
        stride: 2,
        channels: 1,
        in_h: 4,
        in_w: 4,
    };

    let input = Node::new(vec![
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], vec![1, 1, 4, 4]);

    let output = layer.forward(input.clone());

    println!("output: {:?}", output.borrow().data);
    assert_eq!(output.borrow().shape, vec![1, 1, 2, 2]);
    assert_eq!(output.borrow().data, vec![6.0, 8.0, 14.0, 16.0]);

    output.borrow_mut().grad = vec![1.0, 1.0, 1.0, 1.0];
    output.borrow_mut().backward();

    println!("input grad: {:?}", input.borrow().grad);
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    assert_eq!(input.borrow().grad, expected);

    println!("maxpool ok");
}