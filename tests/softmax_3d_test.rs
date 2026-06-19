use rstorch::autograd::graph::softmax::softmax;
use rstorch::autograd::node::{Node, backward_graph};

#[test]
fn softmax_3d_forward() {
    let data = vec![
        1.0, 2.0, 3.0,    0.0, 0.0, 0.0,
        -1.0, 0.0, 1.0,   5.0, 5.0, 5.0,
    ];
    let input = Node::new(data, vec![2, 2, 3]);
    let out = softmax(input);

    let d = out.borrow().data.clone();
    assert_eq!(out.borrow().shape, vec![2, 2, 3]);

    for r in 0..4 {
        let s: f32 = d[r*3..r*3+3].iter().sum();
        println!("row {} sum: {}", r, s);
        assert!((s - 1.0).abs() < 1e-5, "row {} doesn't sum to 1", r);
    }

    assert!((d[3] - 1.0/3.0).abs() < 1e-5);
    assert!((d[4] - 1.0/3.0).abs() < 1e-5);
    assert!((d[5] - 1.0/3.0).abs() < 1e-5);

    assert!((d[9] - 1.0/3.0).abs() < 1e-5);

    println!("softmax 3d forward ok");
}

fn numerical_grad(data: &Vec<f32>, shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; data.len()];
    for i in 0..data.len() {
        let mut plus = data.clone();
        plus[i] += h;
        let out_p = softmax(Node::new(plus, shape.clone()));
        let loss_p: f32 = out_p.borrow().data.iter().enumerate().map(|(j,x)| (j as f32 + 1.0) * x).sum();

        let mut minus = data.clone();
        minus[i] -= h;
        let out_m = softmax(Node::new(minus, shape.clone()));
        let loss_m: f32 = out_m.borrow().data.iter().enumerate().map(|(j,x)| (j as f32 + 1.0) * x).sum();

        grad[i] = (loss_p - loss_m) / (2.0 * h);
    }
    grad
}

#[test]
fn softmax_3d_gradcheck() {
    let shape = vec![2, 2, 3];
    let data = vec![
        0.5, 1.0, 1.5,    0.2, 0.4, 0.6,
        1.0, 0.0, -1.0,   0.3, 0.1, 0.2,
    ];

    let input = Node::new(data.clone(), shape.clone());
    let out = softmax(input.clone());

    let grad_inj: Vec<f32> = (0..out.borrow().data.len()).map(|j| j as f32 + 1.0).collect();
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let analytic = input.borrow().grad.clone();
    let numeric = numerical_grad(&data, &shape);

    println!("analytic: {:?}", analytic);
    println!("numeric:  {:?}", numeric);

    for i in 0..data.len() {
        let diff = (analytic[i] - numeric[i]).abs();
        assert!(diff < 2e-2, "gradient mismatch at {}: {} vs {}", i, analytic[i], numeric[i]);
    }
    println!("softmax 3d gradcheck ok");
}