use rstorch::autograd::node::{Node, backward_graph};
use rstorch::autograd::graph;

#[test]
fn slice_concat_roundtrip() {
    let a = Node::new(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ], vec![2, 6]);

    let left = graph::slice_cols(a.clone(), 0, 3);
    let right = graph::slice_cols(a.clone(), 3, 6);

    println!("left: {:?}", left.borrow().data);
    println!("right: {:?}", right.borrow().data);
    assert_eq!(left.borrow().data, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    assert_eq!(right.borrow().data, vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);

    let joined = graph::concat_cols(vec![left, right]);
    println!("joined: {:?}", joined.borrow().data);
    assert_eq!(joined.borrow().data, a.borrow().data);
    assert_eq!(joined.borrow().shape, vec![2, 6]);

    joined.borrow_mut().grad = vec![1.0; 12];
    backward_graph(&joined);

    println!("a grad: {:?}", a.borrow().grad);
    assert_eq!(a.borrow().grad, vec![1.0; 12]);

    println!("slice/concat roundtrip ok");
}