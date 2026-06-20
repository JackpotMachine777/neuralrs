use rstorch::autograd::graph::relu::relu;
use rstorch::autograd::graph::mul::mul;
use rstorch::autograd::node::{Node, backward_graph};

#[test]
fn relu_parallel_correct() {
    let n = 20000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 10000.0) * 0.001).collect();
    let input = Node::new(data.clone(), vec![n]);
    let out = relu(input.clone());

    let out_data = out.borrow().data.clone();
    for i in 0..n {
        let expected = if data[i] > 0.0 { data[i] } else { 0.0 };
        assert!((out_data[i] - expected).abs() < 1e-7, "forward mismatch at {i}");
    }

    let grad_inj: Vec<f32> = vec![1.0; n];
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let in_grad = input.borrow().grad.clone();
    for i in 0..n {
        let expected = if data[i] > 0.0 { 1.0 } else { 0.0 };
        assert!((in_grad[i] - expected).abs() < 1e-7, "backward mismatch at {i}");
    }
    println!("relu parallel ok (n={n})");
}

#[test]
fn relu_small_sequential_correct() {
    let data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let input = Node::new(data.clone(), vec![5]);
    let out = relu(input.clone());

    let out_data = out.borrow().data.clone();
    let expected = [0.0, 0.0, 0.0, 0.5, 2.0];
    for i in 0..5 {
        assert!((out_data[i] - expected[i]).abs() < 1e-7, "mismatch at {i}");
    }
    println!("relu small sequential ok");
}

#[test]
fn mul_parallel_correct() {
    let n = 20000;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 + 1.0).collect();

    let a = Node::new(a_data.clone(), vec![n]);
    let b = Node::new(b_data.clone(), vec![n]);
    let out = mul(a.clone(), b.clone());

    let out_data = out.borrow().data.clone();
    for i in 0..n {
        assert!((out_data[i] - a_data[i] * b_data[i]).abs() < 1e-4, "forward mismatch at {i}");
    }

    let grad_inj: Vec<f32> = vec![1.0; n];
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let a_grad = a.borrow().grad.clone();
    let b_grad = b.borrow().grad.clone();
    for i in 0..n {
        assert!((a_grad[i] - b_data[i]).abs() < 1e-4, "grad_a mismatch at {i}");
        assert!((b_grad[i] - a_data[i]).abs() < 1e-4, "grad_b mismatch at {i}");
    }
    println!("mul parallel ok (n={n})");
}