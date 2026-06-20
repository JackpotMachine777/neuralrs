use neuralrs::autograd::node::Node;
use neuralrs::autograd::graph::transpose;

#[test]
fn transpose_forward_backward() {
    let a = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let t = transpose::transpose(a.clone());

    println!("transposed: {:?}", t.borrow().data);
    assert_eq!(t.borrow().shape, vec![3, 2]);
    assert_eq!(t.borrow().data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    t.borrow_mut().grad = vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0];
    t.borrow_mut().backward();

    println!("a grad: {:?}", a.borrow().grad);
    assert_eq!(a.borrow().grad, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    println!("transpose ok");
}