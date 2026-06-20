use neuralrs::nn::module::Module;
use neuralrs::nn::dropout2d::Dropout2d;
use neuralrs::autograd::node::{Node, backward_graph};

#[test]
fn dropout2d_eval_passthrough() {
    let mut d = Dropout2d { probability: 0.5, training: false };
    let x = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 2, 2]);
    let out = d.forward(x.clone());
    assert_eq!(out.borrow().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    println!("eval passthrough ok");
}

#[test]
fn dropout2d_drops_all_when_p_one() {
    let mut d = Dropout2d { probability: 1.0, training: true };
    let x = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 2, 2]);
    let out = d.forward(x.clone());
    println!("p=1 output: {:?}", out.borrow().data);
    assert_eq!(out.borrow().data, vec![0.0; 8]);

    out.borrow_mut().grad = vec![1.0; 8];
    backward_graph(&out);
    assert_eq!(x.borrow().grad, vec![0.0; 8]);
    println!("drop all ok");
}

#[test]
fn dropout2d_keeps_all_when_p_zero() {
    let mut d = Dropout2d { probability: 0.0, training: true };
    let x = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 2, 2]);
    let out = d.forward(x.clone());
    println!("p=0 output: {:?}", out.borrow().data);
    assert_eq!(out.borrow().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    out.borrow_mut().grad = vec![1.0; 8];
    backward_graph(&out);
    assert_eq!(x.borrow().grad, vec![1.0; 8]);
    println!("keep all ok");
}

#[test]
fn dropout2d_whole_channel_consistency() {
    let mut d = Dropout2d { probability: 1.0, training: true };
    let x = Node::new((1..=8).map(|v| v as f32).collect(), vec![1, 2, 2, 2]);
    let out = d.forward(x.clone());
    let data = out.borrow().data.clone();

    let ch0_all_zero = data[0..4].iter().all(|&v| v == 0.0);
    let ch1_all_zero = data[4..8].iter().all(|&v| v == 0.0);
    assert!(ch0_all_zero && ch1_all_zero);
    println!("whole channel consistency ok");
}