use rstorch::autograd::node::Node;
use rstorch::autograd::graph;
use rstorch::autograd::engine;

#[test]
fn autograd_softmax_test() {
    let a = Node::new(vec![1.0, 2.0, 3.0], vec![3]);

    let c = graph::softmax(a.clone());

    println!("[SOFTMAX TEST]");
    println!("output: {:?}", c.borrow().data);
    
    let sum: f32 = c.borrow().data.iter().sum();
    println!("sum: {}", sum);
    assert!((sum - 1.0).abs() < 1e-5);

    engine::backward(c.clone());
    println!("a.grad: {:?}", a.borrow().grad);
}