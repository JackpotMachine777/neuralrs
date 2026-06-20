use neuralrs::nn::batchnorm2d::BatchNorm2d;
use neuralrs::nn::module::Module;
use neuralrs::autograd::node::Node;
use neuralrs::tensor::Tensor;

fn numerical_grad(layer: &mut BatchNorm2d, input_data: &Vec<f32>, shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; input_data.len()];

    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += h;
        let out_plus = layer.forward(Node::new(plus, shape.clone()));
        let loss_plus: f32 = out_plus.borrow().data.iter().enumerate().map(|(j, x)| (j as f32 + 1.0) * x).sum();

        let mut minus = input_data.clone();
        minus[i] -= h;
        let out_minus = layer.forward(Node::new(minus, shape.clone()));
        let loss_minus: f32 = out_minus.borrow().data.iter().enumerate().map(|(j, x)| (j as f32 + 1.0) * x).sum();

        grad[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    grad
}

#[test]
fn gradcheck_batchnorm2d() {
    let make_layer = || BatchNorm2d {
        gamma: Tensor::new(vec![1.0, 2.0], vec![2]),
        beta: Tensor::new(vec![0.0, 0.0], vec![2]),
        epsilon: 1e-5,
        num_channels: 2,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        running_mean: vec![0.0; 2],
        running_var: vec![1.0; 2],
        momentum: 0.9,
        training: true,
    };

    let input_data = vec![
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 0.5, 1.5,
        0.5, 1.5, 2.5, 3.5,
        3.0, 2.0, 1.0, 0.0,
    ];
    let shape = vec![2, 2, 2, 2];

    let mut layer_n = make_layer();
    let numeric = numerical_grad(&mut layer_n, &input_data, &shape);

    let mut layer_a = make_layer();
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer_a.forward(input.clone());

    let grad_inj: Vec<f32> = (0..output.borrow().data.len()).map(|j| j as f32 + 1.0).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();

    let analytic = input.borrow().grad.clone();

    println!("numeric:  {numeric:?}");
    println!("analytic: {analytic:?}");

    for i in 0..numeric.len() {
        let diff = (numeric[i] - analytic[i]).abs();
        println!("diff[{i}] = {diff}");
        assert!(diff < 2e-2, "gradient mismatch at {i}");
    }
}

#[test]
fn batchnorm2d_eval_single_sample() {
    let mut bn = BatchNorm2d {
        gamma: Tensor::new(vec![1.0, 1.0], vec![2]),
        beta: Tensor::new(vec![0.0, 0.0], vec![2]),
        epsilon: 1e-5,
        num_channels: 2,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        running_mean: vec![0.0; 2],
        running_var: vec![1.0; 2],
        momentum: 0.9,
        training: true,
    };

    for _ in 0..5 {
        let input = Node::new(
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.5, 1.5,
                 0.5, 1.5, 2.5, 3.5, 3.0, 2.0, 1.0, 0.0],
            vec![2, 2, 2, 2],
        );
        let _ = bn.forward(input);
    }

    bn.set_training(false);
    let single = Node::new(vec![1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.5, 1.5], vec![1, 2, 2, 2]);
    let out = bn.forward(single);

    let data = out.borrow().data.clone();
    println!("eval output (N=1): {data:?}");
    assert!(data.iter().all(|x| x.is_finite()), "non-finite values in eval!");
    assert_eq!(out.borrow().shape, vec![1, 2, 2, 2]);

    println!("batchnorm2d eval single sample ok");
}