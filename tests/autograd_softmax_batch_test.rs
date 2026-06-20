use neuralrs::autograd::node::Node;
use neuralrs::autograd::graph::softmax;

#[test]
fn softmax_batch_test() {
    let a = Node::new(vec![
        1.0, 2.0, 3.0,
        10.0, 10.0, 10.0,
    ], vec![2, 3]);

    let out = softmax::softmax(a.clone());
    let data = out.borrow().data.clone();

    println!("output: {data:?}");

    let row0_sum: f32 = data[0..3].iter().sum();
    let row1_sum: f32 = data[3..6].iter().sum();
    println!("row0 sum: {row0_sum}, row1 sum: {row1_sum}");

    assert!((row0_sum - 1.0).abs() < 1e-5);
    assert!((row1_sum - 1.0).abs() < 1e-5);

    assert!((data[3] - 0.3333).abs() < 1e-3);
    assert!((data[4] - 0.3333).abs() < 1e-3);
    assert!((data[5] - 0.3333).abs() < 1e-3);

    println!("softmax batch ok");
}