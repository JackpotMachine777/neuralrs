use rstorch::tensor::Tensor;
use rstorch::nn::multihead::MultiHeadAttention;
use rstorch::autograd::node::{Node, backward_graph};
use rstorch::init::xavier;

#[test]
fn multihead_forward_backward() {
    let seq_len = 4;
    let d_model = 8;
    let num_heads = 2;

    let mut mha = MultiHeadAttention {
        w_q: Tensor::new(xavier::xavier(d_model, d_model), vec![d_model, d_model]),
        w_k: Tensor::new(xavier::xavier(d_model, d_model), vec![d_model, d_model]),
        w_v: Tensor::new(xavier::xavier(d_model, d_model), vec![d_model, d_model]),
        w_o: Tensor::new(xavier::xavier(d_model, d_model), vec![d_model, d_model]),
        d_model,
        num_heads,
        w_q_node: None,
        w_k_node: None,
        w_v_node: None,
        w_o_node: None,
    };
    mha.zero_grad();

    let x = Node::new(
        (0..seq_len*d_model).map(|i| (i as f32 % 7.0) * 0.03 - 0.1).collect(),
        vec![seq_len, d_model],
    );

    let out = mha.forward(x.clone());

    println!("out shape: {:?}", out.borrow().shape);
    assert_eq!(out.borrow().shape, vec![seq_len, d_model]);

    out.borrow_mut().grad = vec![1.0; seq_len * d_model];
    backward_graph(&out);

    mha.sync_grads();

    let wq: f32 = mha.w_q.grad.iter().map(|v| v.abs()).sum();
    let wk: f32 = mha.w_k.grad.iter().map(|v| v.abs()).sum();
    let wv: f32 = mha.w_v.grad.iter().map(|v| v.abs()).sum();
    let wo: f32 = mha.w_o.grad.iter().map(|v| v.abs()).sum();
    println!("w_q: {}, w_k: {}, w_v: {}, w_o: {}", wq, wk, wv, wo);

    assert!(wo > 0.0, "W_o gradient zero!");
    assert!(wv > 0.0, "W_v gradient zero!");

    let x_grad: f32 = x.borrow().grad.iter().map(|v| v.abs()).sum();
    println!("x grad sum: {}", x_grad);
    assert!(x_grad > 0.0, "x gradient zero - rozgaleziony backward nie dziala!");

    println!("multi-head attention ok");
}