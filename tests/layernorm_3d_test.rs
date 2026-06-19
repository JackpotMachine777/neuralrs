use rstorch::nn::normalization::LayerNorm;
use rstorch::nn::module::Module;
use rstorch::autograd::node::{Node, backward_graph};
use rstorch::tensor::Tensor;

fn make_ln(features: usize) -> LayerNorm {
    LayerNorm {
        gamma: Tensor::new(vec![1.0; features], vec![features]),
        beta: Tensor::new(vec![0.0; features], vec![features]),
        epsilon: 1e-5,
        num_features: features,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; features])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; features])),
    }
}

#[test]
fn layernorm_3d_forward() {
    let mut ln = make_ln(3);
    let data = vec![
        1.0, 2.0, 3.0,    4.0, 5.0, 6.0,
        0.0, 0.0, 0.0,    -1.0, 0.0, 1.0,
    ];
    let input = Node::new(data, vec![2, 2, 3]);
    let out = ln.forward(input);

    assert_eq!(out.borrow().shape, vec![2, 2, 3]);
    let d = out.borrow().data.clone();

    for r in 0..4 {
        let m: f32 = d[r*3..r*3+3].iter().sum::<f32>() / 3.0;
        println!("token {} mean: {}", r, m);
        assert!(m.abs() < 1e-4, "token {} not zero-mean", r);
    }
    println!("layernorm 3d forward ok");
}

fn numerical_grad(ln: &mut LayerNorm, data: &Vec<f32>, shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; data.len()];
    for i in 0..data.len() {
        let mut plus = data.clone();
        plus[i] += h;
        let out_p = ln.forward(Node::new(plus, shape.clone()));
        let loss_p: f32 = out_p.borrow().data.iter().enumerate().map(|(j,x)| (j as f32 + 1.0) * x).sum();

        let mut minus = data.clone();
        minus[i] -= h;
        let out_m = ln.forward(Node::new(minus, shape.clone()));
        let loss_m: f32 = out_m.borrow().data.iter().enumerate().map(|(j,x)| (j as f32 + 1.0) * x).sum();

        grad[i] = (loss_p - loss_m) / (2.0 * h);
    }
    grad
}

#[test]
fn layernorm_3d_gradcheck() {
    let shape = vec![2, 2, 3];
    let data = vec![
        0.5, 1.0, 1.5,   0.2, 0.8, 0.4,
        1.0, 0.0, 2.0,   0.3, 0.6, 0.1,
    ];

    let mut ln_a = make_ln(3);
    let input = Node::new(data.clone(), shape.clone());
    let out = ln_a.forward(input.clone());

    let grad_inj: Vec<f32> = (0..out.borrow().data.len()).map(|j| j as f32 + 1.0).collect();
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let analytic = input.borrow().grad.clone();

    let mut ln_n = make_ln(3);
    let numeric = numerical_grad(&mut ln_n, &data, &shape);

    println!("analytic: {:?}", analytic);
    println!("numeric:  {:?}", numeric);

    for i in 0..data.len() {
        let diff = (analytic[i] - numeric[i]).abs();
        assert!(diff < 2e-2, "gradient mismatch at {}: {} vs {}", i, analytic[i], numeric[i]);
    }
    println!("layernorm 3d gradcheck ok");
}