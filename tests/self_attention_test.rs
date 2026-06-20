use rstorch::tensor::Tensor;
use rstorch::nn::self_attention::SelfAttention;
use rstorch::autograd::node::{Node, backward_graph};
use rstorch::init::xavier;

#[test]
fn self_attention_forward_backward() {
    let seq_len = 3;
    let d_model = 4;
    let d_k = 4;

    let mut layer = SelfAttention {
        w_q: Tensor::new(xavier::xavier(d_model, d_k), vec![d_model, d_k]),
        w_k: Tensor::new(xavier::xavier(d_model, d_k), vec![d_model, d_k]),
        w_v: Tensor::new(xavier::xavier(d_model, d_k), vec![d_model, d_k]),
        d_model,
        d_k,
        w_q_node: None,
        w_k_node: None,
        w_v_node: None,
    };
    layer.zero_grad();

    let x = Node::new(
        (0..seq_len*d_model).map(|i| (i as f32 % 5.0) * 0.05 - 0.1).collect(),
        vec![seq_len, d_model],
    );

    let out = layer.forward(x.clone());

    println!("out shape: {:?}", out.borrow().shape);
    assert_eq!(out.borrow().shape, vec![seq_len, d_k]);

    out.borrow_mut().grad = vec![1.0; seq_len * d_k];
    backward_graph(&out);

    layer.sync_grads();

    let wq: f32 = layer.w_q.grad.iter().map(|x| x.abs()).sum();
    let wk: f32 = layer.w_k.grad.iter().map(|x| x.abs()).sum();
    let wv: f32 = layer.w_v.grad.iter().map(|x| x.abs()).sum();
    println!("w_q: {wq}, w_k: {wk}, w_v: {wv}");

    assert!(wv > 0.0, "W_v gradient zero!");

    let x_grad: f32 = x.borrow().grad.iter().map(|v| v.abs()).sum();
    println!("x grad sum: {x_grad}");
    assert!(x_grad > 0.0, "x gradient zero - topo backward through 3 layers doesnt work!");

    println!("self-attention ok");
}