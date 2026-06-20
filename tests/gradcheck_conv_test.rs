use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::conv::Conv2d;
use rstorch::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

fn make_layer() -> Conv2d {
    Conv2d {
        weight: Tensor::new(vec![
            0.5, -0.3, 0.2, 0.1,
            0.4, 0.6, -0.1, 0.3,
        ], vec![1, 2, 2, 2]),
        bias: Tensor::new(vec![0.0], vec![1]),
        c_in: 2,
        c_out: 1,
        kh: 2,
        kw: 2,
        stride: 1,
        in_h: 3,
        in_w: 3,
        weight_grad: Rc::new(RefCell::new(vec![0.0; 8])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; 1])),
        padding: 0,
    }
}

fn loss_of(out: &Vec<f32>) -> f32 {
    out.iter().enumerate().map(|(i, x)| (i as f32 + 1.0) * x).sum()
}

#[test]
fn gradcheck_conv_input() {
    let input_data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        0.5, 1.5, 2.5,
        3.5, 4.5, 5.5,
        6.5, 7.5, 8.5,
    ];
    let shape = vec![1, 2, 3, 3];

    let h = 1e-3;
    let mut numeric = vec![0.0; input_data.len()];
    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += h;
        let mut lp = make_layer();
        let op = lp.forward(Node::new(plus, shape.clone()));
        let loss_plus = loss_of(&op.borrow().data);

        let mut minus = input_data.clone();
        minus[i] -= h;
        let mut lm = make_layer();
        let om = lm.forward(Node::new(minus, shape.clone()));
        let loss_minus = loss_of(&om.borrow().data);

        numeric[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    let mut layer = make_layer();
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer.forward(input.clone());
    let out_len = output.borrow().data.len();
    let grad_inj: Vec<f32> = (0..out_len).map(|i| i as f32 + 1.0).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();
    let analytic = input.borrow().grad.clone();

    println!("numeric:  {numeric:?}");
    println!("analytic: {analytic:?}");

    for i in 0..numeric.len() {
        let diff = (numeric[i] - analytic[i]).abs();
        assert!(diff < 1e-2, "input grad mismatch at {}: {} vs {}", i, numeric[i], analytic[i]);
    }
}

#[test]
fn gradcheck_conv_weight() {
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
    ];
    let shape = vec![1, 2, 3, 3];

    let base = make_layer();
    let weight_len = base.weight.storage.data.len();

    let h = 1e-3;
    let mut numeric = vec![0.0; weight_len];
    for i in 0..weight_len {
        let mut lp = make_layer();
        lp.weight.storage.data[i] += h;
        let op = lp.forward(Node::new(input_data.clone(), shape.clone()));
        let loss_plus = loss_of(&op.borrow().data);

        let mut lm = make_layer();
        lm.weight.storage.data[i] -= h;
        let om = lm.forward(Node::new(input_data.clone(), shape.clone()));
        let loss_minus = loss_of(&om.borrow().data);

        numeric[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    let mut layer = make_layer();
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer.forward(input.clone());
    let out_len = output.borrow().data.len();
    let grad_inj: Vec<f32> = (0..out_len).map(|i| i as f32 + 1.0).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();
    layer.sync_grads();
    let analytic = layer.weight.grad.clone();

    println!("weight numeric:  {numeric:?}");
    println!("weight analytic: {analytic:?}");

    for i in 0..numeric.len() {
        let diff = (numeric[i] - analytic[i]).abs();
        assert!(diff < 1e-2, "weight grad mismatch at {}: {} vs {}", i, numeric[i], analytic[i]);
    }
}

#[test]
fn gradcheck_conv_bias() {
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
    ];
    let shape = vec![1, 2, 3, 3];

    let h = 1e-3;
    let mut numeric = vec![0.0; 1];
    {
        let mut lp = make_layer();
        lp.bias.storage.data[0] += h;
        let op = lp.forward(Node::new(input_data.clone(), shape.clone()));
        let loss_plus = loss_of(&op.borrow().data);

        let mut lm = make_layer();
        lm.bias.storage.data[0] -= h;
        let om = lm.forward(Node::new(input_data.clone(), shape.clone()));
        let loss_minus = loss_of(&om.borrow().data);

        numeric[0] = (loss_plus - loss_minus) / (2.0 * h);
    }

    let mut layer = make_layer();
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer.forward(input.clone());
    let out_len = output.borrow().data.len();
    let grad_inj: Vec<f32> = (0..out_len).map(|i| i as f32 + 1.0).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();
    layer.sync_grads();
    let analytic = layer.bias.grad.clone();

    println!("bias numeric:  {numeric:?}");
    println!("bias analytic: {analytic:?}");

    let diff = (numeric[0] - analytic[0]).abs();
    assert!(diff < 1e-2, "bias grad mismatch: {} vs {}", numeric[0], analytic[0]);
}

#[test]
fn gradcheck_conv_padding() {
    let make_layer = || Conv2d {
        weight: Tensor::new(vec![
            0.5, -0.3, 0.2, 0.1,
            0.4, 0.6, -0.1, 0.3,
        ], vec![1, 2, 2, 2]),
        bias: Tensor::new(vec![0.0], vec![1]),
        c_in: 2, c_out: 1, kh: 2, kw: 2, stride: 1,
        padding: 1,
        in_h: 3, in_w: 3,
        weight_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 8])),
        bias_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 1])),
    };

    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
    ];
    let shape = vec![1, 2, 3, 3];

    let h = 1e-3;
    let mut numeric = vec![0.0; input_data.len()];
    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += h;
        let mut lp = make_layer();
        let op = lp.forward(Node::new(plus, shape.clone()));
        let loss_plus = loss_of(&op.borrow().data);

        let mut minus = input_data.clone();
        minus[i] -= h;
        let mut lm = make_layer();
        let om = lm.forward(Node::new(minus, shape.clone()));
        let loss_minus = loss_of(&om.borrow().data);

        numeric[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    let mut layer = make_layer();
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer.forward(input.clone());
    let out_len = output.borrow().data.len();
    let grad_inj: Vec<f32> = (0..out_len).map(|i| i as f32 + 1.0).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();
    let analytic = input.borrow().grad.clone();

    println!("pad numeric:  {numeric:?}");
    println!("pad analytic: {analytic:?}");

    for i in 0..numeric.len() {
        let diff = (numeric[i] - analytic[i]).abs();
        assert!(diff < 5e-2, "padded input grad mismatch at {i}");
    }
    println!("conv padding gradcheck ok");
}