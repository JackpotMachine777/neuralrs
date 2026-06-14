use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::conv::Conv2d;
use rstorch::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

#[test]
fn conv2d_padding_test() {
    let mut layer = Conv2d {
        weight: Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]),
        bias: Tensor::new(vec![0.0], vec![1]),
        c_in: 1, c_out: 1, kh: 3, kw: 3, stride: 1,
        padding: 1,
        in_h: 3, in_w: 3,
        weight_grad: Rc::new(RefCell::new(vec![0.0; 9])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; 1])),
    };

    let input = Node::new(vec![1.0; 9], vec![1, 3, 3]);

    let output = layer.forward(input);

    println!("output: {:?}", output.borrow().data);
    println!("shape:  {:?}", output.borrow().shape);

    assert_eq!(output.borrow().shape, vec![1, 3, 3]);
    assert_eq!(output.borrow().data, vec![
        4.0, 6.0, 4.0,
        6.0, 9.0, 6.0,
        4.0, 6.0, 4.0,
    ]);

    println!("padding ok");
}