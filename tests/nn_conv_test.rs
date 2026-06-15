use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::conv::Conv2d;
use rstorch::autograd::node::Node;

#[test]
fn conv2d_forward_test() {
    let mut layer = Conv2d {
        weight: Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]),
        bias: Tensor::new(vec![0.0], vec![1]),
        c_in: 1,
        c_out: 1,
        kh: 2,
        kw: 2,
        stride: 1,
        in_h: 3,
        in_w: 3,
        weight_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 4])),
        bias_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 1])),
        padding: 0,
    };

    let input = Node::new(vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], vec![1, 1, 3, 3]);

    let output = layer.forward(input);

    println!("output: {:?}", output.borrow().data);
    println!("shape:  {:?}", output.borrow().shape);

    assert_eq!(output.borrow().shape, vec![1, 1, 2, 2]);
    assert_eq!(output.borrow().data, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn conv2d_multichannel_test() {
    let mut layer = Conv2d {
        weight: Tensor::new(vec![
            1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,  2.0, 2.0, 2.0, 2.0,
        ], vec![2, 2, 2, 2]),
        bias: Tensor::new(vec![0.0, 0.0], vec![2]),
        c_in: 2,
        c_out: 2,
        kh: 2,
        kw: 2,
        stride: 1,
        in_h: 2,
        in_w: 2,
        weight_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 16])),
        bias_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        padding: 0,
    };

    let input = Node::new(vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 
    ], vec![1, 2, 2, 2]);

    let output = layer.forward(input);

    println!("output: {:?}", output.borrow().data);
    println!("shape:  {:?}", output.borrow().shape);

    assert_eq!(output.borrow().shape, vec![1, 2, 1, 1]);
    assert_eq!(output.borrow().data, vec![36.0, 72.0]);
}