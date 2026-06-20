use neuralrs::autograd::graph::bmm::bmm;
use neuralrs::autograd::node::{Node, backward_graph};

fn numerical_grad_a(a_data: &Vec<f32>, a_shape: &Vec<usize>, b_data: &Vec<f32>, b_shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; a_data.len()];
    for i in 0..a_data.len() {
        let mut plus = a_data.clone();
        plus[i] += h;
        let out_p = bmm(Node::new(plus, a_shape.clone()), Node::new(b_data.clone(), b_shape.clone()));
        let loss_p: f32 = out_p.borrow().data.iter().sum();

        let mut minus = a_data.clone();
        minus[i] -= h;
        let out_m = bmm(Node::new(minus, a_shape.clone()), Node::new(b_data.clone(), b_shape.clone()));
        let loss_m: f32 = out_m.borrow().data.iter().sum();

        grad[i] = (loss_p - loss_m) / (2.0 * h);
    }
    grad
}

fn numerical_grad_b(a_data: &Vec<f32>, a_shape: &Vec<usize>, b_data: &Vec<f32>, b_shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; b_data.len()];
    for i in 0..b_data.len() {
        let mut plus = b_data.clone();
        plus[i] += h;
        let out_p = bmm(Node::new(a_data.clone(), a_shape.clone()), Node::new(plus, b_shape.clone()));
        let loss_p: f32 = out_p.borrow().data.iter().sum();

        let mut minus = b_data.clone();
        minus[i] -= h;
        let out_m = bmm(Node::new(a_data.clone(), a_shape.clone()), Node::new(minus, b_shape.clone()));
        let loss_m: f32 = out_m.borrow().data.iter().sum();

        grad[i] = (loss_p - loss_m) / (2.0 * h);
    }
    grad
}

#[test]
fn gradcheck_bmm() {
    let a_shape = vec![2, 2, 3];
    let b_shape = vec![2, 3, 2];

    let a_data = vec![
        1.0, 2.0, 3.0,   4.0, 5.0, 6.0,
        0.5, 1.5, 2.5,   3.5, 4.5, 5.5,
    ];
    let b_data = vec![
        1.0, 2.0,   3.0, 4.0,   5.0, 6.0,
        0.1, 0.2,   0.3, 0.4,   0.5, 0.6,
    ];

    let a = Node::new(a_data.clone(), a_shape.clone());
    let b = Node::new(b_data.clone(), b_shape.clone());
    let out = bmm(a.clone(), b.clone());

    let grad_inj: Vec<f32> = vec![1.0; out.borrow().data.len()];
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let a_analytic = a.borrow().grad.clone();
    let b_analytic = b.borrow().grad.clone();

    let a_numeric = numerical_grad_a(&a_data, &a_shape, &b_data, &b_shape);
    let b_numeric = numerical_grad_b(&a_data, &a_shape, &b_data, &b_shape);

    println!("A analytic: {a_analytic:?}");
    println!("A numeric:  {a_numeric:?}");
    println!("B analytic: {b_analytic:?}");
    println!("B numeric:  {b_numeric:?}");

    for i in 0..a_data.len() {
        let diff = (a_analytic[i] - a_numeric[i]).abs();
        assert!(diff < 2e-2, "A gradient mismatch at {}: {} vs {}", i, a_analytic[i], a_numeric[i]);
    }
    for i in 0..b_data.len() {
        let diff = (b_analytic[i] - b_numeric[i]).abs();
        assert!(diff < 2e-2, "B gradient mismatch at {}: {} vs {}", i, b_analytic[i], b_numeric[i]);
    }

    println!("bmm gradcheck ok");
}

#[test]
fn bmm_forward_correct() {
    let a = Node::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    let b = Node::new(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);
    let out = bmm(a, b);

    let data = out.borrow().data.clone();
    println!("forward: {data:?}");
    assert_eq!(out.borrow().shape, vec![1, 2, 2]);
    assert!((data[0] - 19.0).abs() < 1e-4);
    assert!((data[1] - 22.0).abs() < 1e-4);
    assert!((data[2] - 43.0).abs() < 1e-4);
    assert!((data[3] - 50.0).abs() < 1e-4);

    println!("bmm forward ok");
}