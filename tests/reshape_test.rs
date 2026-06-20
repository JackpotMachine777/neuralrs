use neuralrs::autograd::graph::reshape::reshape;
use neuralrs::autograd::node::{Node, backward_graph};

#[test]
fn reshape_forward() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Node::new(data.clone(), vec![2, 3]);
    let out = reshape(input, vec![3, 2]);

    assert_eq!(out.borrow().shape, vec![3, 2]);
    assert_eq!(out.borrow().data, data);
    println!("reshape forward ok");
}

#[test]
fn reshape_3d_to_2d() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let input = Node::new(data.clone(), vec![2, 2, 3]);
    let out = reshape(input, vec![4, 3]);

    assert_eq!(out.borrow().shape, vec![4, 3]);
    assert_eq!(out.borrow().data, data);
    println!("reshape 3d->2d ok");
}

#[test]
fn reshape_gradcheck() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Node::new(data.clone(), vec![2, 3]);
    let out = reshape(input.clone(), vec![3, 2]);

    let grad_inj: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    out.borrow_mut().grad = grad_inj.clone();
    backward_graph(&out);

    let in_grad = input.borrow().grad.clone();
    println!("input grad: {in_grad:?}");

    for i in 0..data.len() {
        assert!((in_grad[i] - grad_inj[i]).abs() < 1e-6, "grad mismatch at {i}");
    }
    println!("reshape gradcheck ok");
}