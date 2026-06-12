use rstorch::autograd::{node::Node, graph, engine};

#[test]
fn autograd_add_test() {
    let a = Node::new(vec![2.0, 3.0], vec![2]);
    let b = Node::new(vec![4.0, 5.0], vec![2]);

    let c = graph::add(a.clone(), b.clone());

    engine::backward(c);

    println!("[ADD TEST]");
    println!("a.grad: {:?}", a.borrow().grad);
    println!("b.grad: {:?}", b.borrow().grad);
    
    assert_eq!(a.borrow().grad, vec![1.0, 1.0]);
    assert_eq!(b.borrow().grad, vec![1.0, 1.0]);
}

#[test]
fn autograd_mul_test() {
    let a = Node::new(vec![2.0, 3.0], vec![2]);
    let b = Node::new(vec![4.0, 5.0], vec![2]);
    
    let c = graph::mul(a.clone(), b.clone());
    
    engine::backward(c);
    
    println!("[MUL TEST]");
    println!("a.grad: {:?}", a.borrow().grad);
    println!("b.grad: {:?}", b.borrow().grad);
    
    assert_eq!(a.borrow().grad, vec![4.0, 5.0]);
    assert_eq!(b.borrow().grad, vec![2.0, 3.0]);
}

#[test]
fn autograd_relu_test() {
    let a = Node::new(vec![2.0, -3.0], vec![2]);
    
    let c = graph::relu(a.clone());
    
    engine::backward(c);
    
    println!("[RELU TEST]");
    println!("a.grad: {:?}", a.borrow().grad);
    
    assert_eq!(a.borrow().grad, vec![1.0, 0.0]);
}