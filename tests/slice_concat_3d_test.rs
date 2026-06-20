use rstorch::autograd::graph::slice_cols::slice_cols;
use rstorch::autograd::graph::concat_cols::concat_cols;
use rstorch::autograd::node::{Node, backward_graph};

#[test]
fn slice_cols_3d() {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Node::new(data, vec![2, 2, 4]);
    let out = slice_cols(input, 1, 3);

    assert_eq!(out.borrow().shape, vec![2, 2, 2]);
    let d = out.borrow().data.clone();
    println!("sliced: {d:?}");

    let expected = [1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0];
    for i in 0..8 {
        assert!((d[i] - expected[i]).abs() < 1e-6, "mismatch at {i}");
    }
    println!("slice_cols 3d ok");
}

#[test]
fn concat_cols_3d() {
    let a = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let b = Node::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], vec![2, 2, 2]);
    let out = concat_cols(vec![a, b]);

    assert_eq!(out.borrow().shape, vec![2, 2, 4]);
    let d = out.borrow().data.clone();
    println!("concat: {d:?}");

    let expected = [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0];
    for i in 0..16 {
        assert!((d[i] - expected[i]).abs() < 1e-6, "mismatch at {i}");
    }
    println!("concat_cols 3d ok");
}

#[test]
fn slice_concat_3d_roundtrip_grad() {
    let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let input = Node::new(data.clone(), vec![2, 2, 4]);

    let left = slice_cols(input.clone(), 0, 2);
    let right = slice_cols(input.clone(), 2, 4);
    let recon = concat_cols(vec![left, right]);

    let grad_inj: Vec<f32> = (0..16).map(|i| (i as f32) + 1.0).collect();
    recon.borrow_mut().grad = grad_inj.clone();
    backward_graph(&recon);

    let in_grad = input.borrow().grad.clone();
    println!("input grad: {in_grad:?}");

    for i in 0..16 {
        assert!((in_grad[i] - grad_inj[i]).abs() < 1e-6, "grad mismatch at {i}");
    }
    println!("slice/concat 3d roundtrip grad ok");
}