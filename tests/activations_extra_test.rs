use rstorch::autograd::node::{Node, backward_graph};
use rstorch::autograd::graph;

fn num_grad_unary<F>(input: &[f32], f: F) -> Vec<f32>
where F: Fn(&[f32]) -> Vec<f32> {
    let eps = 1e-3;
    let mut grad = vec![0.0; input.len()];
    for i in 0..input.len() {
        let mut plus = input.to_vec();
        let mut minus = input.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let fp: f32 = f(&plus).iter().sum();
        let fm: f32 = f(&minus).iter().sum();
        grad[i] = (fp - fm) / (2.0 * eps);
    }
    grad
}

#[test]
fn gradcheck_leaky_relu() {
    let data = vec![1.5, -2.0, 0.5, -0.8];
    let alpha = 0.01;
    let a = Node::new(data.clone(), vec![4]);
    let out = graph::leaky_relu(a.clone(), alpha);
    out.borrow_mut().grad = vec![1.0; 4];
    backward_graph(&out);
    let analytic = a.borrow().grad.clone();

    let numeric = num_grad_unary(&data, |x| {
        x.iter().map(|&v| if v > 0.0 { v } else { alpha * v }).collect()
    });

    println!("leaky analytic: {analytic:?}");
    println!("leaky numeric:  {numeric:?}");
    for i in 0..4 {
        assert!((analytic[i] - numeric[i]).abs() < 1e-2, "leaky mismatch at {i}");
    }
    println!("leaky_relu ok");
}

#[test]
fn gradcheck_elu() {
    let data = vec![1.5, -2.0, 0.5, -0.8];
    let alpha = 1.0;
    let a = Node::new(data.clone(), vec![4]);
    let out = graph::elu(a.clone(), alpha);
    out.borrow_mut().grad = vec![1.0; 4];
    backward_graph(&out);
    let analytic = a.borrow().grad.clone();

    let numeric = num_grad_unary(&data, |x| {
        x.iter().map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }).collect()
    });

    println!("elu analytic: {analytic:?}");
    println!("elu numeric:  {numeric:?}");
    for i in 0..4 {
        assert!((analytic[i] - numeric[i]).abs() < 1e-2, "elu mismatch at {i}");
    }
    println!("elu ok");
}

#[test]
fn gradcheck_silu() {
    let data = vec![1.5, -2.0, 0.5, -0.8];
    let a = Node::new(data.clone(), vec![4]);
    let out = graph::silu(a.clone());
    out.borrow_mut().grad = vec![1.0; 4];
    backward_graph(&out);
    let analytic = a.borrow().grad.clone();

    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let numeric = num_grad_unary(&data, |x| {
        x.iter().map(|&v| v * sig(v)).collect()
    });

    println!("silu analytic: {analytic:?}");
    println!("silu numeric:  {numeric:?}");
    for i in 0..4 {
        assert!((analytic[i] - numeric[i]).abs() < 1e-2, "silu mismatch at {i}");
    }
    println!("silu ok");
}