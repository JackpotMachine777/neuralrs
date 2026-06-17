use rstorch::nn::positional::PositionalEncoding;
use rstorch::autograd::node::{Node, backward_graph};

#[test]
fn positional_encoding_values() {
    let d_model = 4;
    let max_len = 10;
    let pe = PositionalEncoding::new(d_model, max_len);

    let x = Node::new(vec![0.0; 3 * d_model], vec![3, d_model]);
    let out = pe.forward(x.clone());

    let data = out.borrow().data.clone();
    println!("PE output: {:?}", data);
    assert_eq!(out.borrow().shape, vec![3, d_model]);

    assert!((data[0] - 0.0).abs() < 1e-5, "pos0 dim0 should be sin(0)=0");
    assert!((data[1] - 1.0).abs() < 1e-5, "pos0 dim1 should be cos(0)=1");
    assert!((data[2] - 0.0).abs() < 1e-5, "pos0 dim2 should be sin(0)=0");
    assert!((data[3] - 1.0).abs() < 1e-5, "pos0 dim3 should be cos(0)=1");

    assert!((data[4] - 1.0_f32.sin()).abs() < 1e-4, "pos1 dim0 should be sin(1)");

    println!("PE values ok");
}

#[test]
fn positional_gradient_flows() {
    let d_model = 4;
    let pe = PositionalEncoding::new(d_model, 10);

    let x = Node::new(vec![0.5; 2 * d_model], vec![2, d_model]);
    let out = pe.forward(x.clone());

    out.borrow_mut().grad = vec![1.0; 2 * d_model];
    backward_graph(&out);

    println!("x grad: {:?}", x.borrow().grad);
    assert_eq!(x.borrow().grad, vec![1.0; 2 * d_model]);

    println!("PE gradient ok");
}