use neuralrs::tensor::Tensor;
use neuralrs::nn::rnn::RNNCell;
use neuralrs::autograd::node::Node;
use neuralrs::init::xavier;

#[test]
fn rnn_sequence_forward_backward() {
    let input_size = 3;
    let hidden_size = 4;

    let mut cell = RNNCell {
        w_xh: Tensor::new(xavier::xavier(input_size, hidden_size), vec![input_size, hidden_size]),
        w_hh: Tensor::new(xavier::xavier(hidden_size, hidden_size), vec![hidden_size, hidden_size]),
        bias: Tensor::new(vec![0.0; hidden_size], vec![hidden_size]),
        input_size,
        hidden_size,
        w_xh_node: None,
        w_hh_node: None,
        bias_node: None,
    };

    cell.zero_grad();

    let seq = vec![
        Node::new(vec![1.0, 0.5, -0.2], vec![1, 3]),
        Node::new(vec![0.3, -0.1, 0.8], vec![1, 3]),
        Node::new(vec![-0.5, 0.2, 0.4], vec![1, 3]),
    ];

    let mut h = Node::new(vec![0.0; hidden_size], vec![1, hidden_size]);

    for x in &seq {
        h = cell.step(x.clone(), h.clone());
    }

    println!("final h: {:?}", h.borrow().data);
    assert_eq!(h.borrow().shape, vec![1, hidden_size]);

    h.borrow_mut().grad = vec![1.0; hidden_size];
    h.borrow_mut().backward();

    cell.sync_grads();

    let w_hh_grad_sum: f32 = cell.w_hh.grad.iter().map(|x| x.abs()).sum();
    println!("w_hh grad sum: {w_hh_grad_sum}");
    assert!(w_hh_grad_sum > 0.0, "w_hh gradient zero - BPTT doesnt work!");

    let w_xh_grad_sum: f32 = cell.w_xh.grad.iter().map(|x| x.abs()).sum();
    println!("w_xh grad sum: {w_xh_grad_sum}");
    assert!(w_xh_grad_sum > 0.0, "w_xh gradient zero!");

    println!("rnn BPTT ok");
}