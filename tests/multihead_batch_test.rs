use neuralrs::nn::multihead::MultiHeadAttention;
use neuralrs::autograd::node::{Node, backward_graph};
use neuralrs::tensor::Tensor;

fn make_mha(d_model: usize, num_heads: usize) -> MultiHeadAttention {
    let mk = |seed: f32| -> Vec<f32> {
        (0..d_model*d_model).map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.1).collect()
    };
    MultiHeadAttention {
        w_q: Tensor::new(mk(1.0), vec![d_model, d_model]),
        w_k: Tensor::new(mk(2.0), vec![d_model, d_model]),
        w_v: Tensor::new(mk(3.0), vec![d_model, d_model]),
        w_o: Tensor::new(mk(4.0), vec![d_model, d_model]),
        d_model,
        num_heads,
        w_q_node: None,
        w_k_node: None,
        w_v_node: None,
        w_o_node: None,
    }
}

#[test]
fn multihead_batch_shape() {
    let mut mha = make_mha(4, 2);
    let n = 2 * 3 * 4;
    let x = Node::new((0..n).map(|i| (i as f32) * 0.01).collect(), vec![2, 3, 4]);
    let out = mha.forward_batch(x);

    assert_eq!(out.borrow().shape, vec![2, 3, 4]);
    assert!(out.borrow().data.iter().all(|v| v.is_finite()));
    println!("multihead batch shape ok");
}

fn numerical_grad_x(mha_builder: &dyn Fn() -> MultiHeadAttention, x_data: &Vec<f32>, shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; x_data.len()];
    for i in 0..x_data.len() {
        let mut plus = x_data.clone();
        plus[i] += h;
        let mut m_p = mha_builder();
        let out_p = m_p.forward_batch(Node::new(plus, shape.clone()));
        let loss_p: f32 = out_p.borrow().data.iter().sum();

        let mut minus = x_data.clone();
        minus[i] -= h;
        let mut m_m = mha_builder();
        let out_m = m_m.forward_batch(Node::new(minus, shape.clone()));
        let loss_m: f32 = out_m.borrow().data.iter().sum();

        grad[i] = (loss_p - loss_m) / (2.0 * h);
    }
    grad
}

#[test]
fn multihead_batch_gradcheck_input() {
    let shape = vec![2, 2, 4];
    let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();

    let builder = || make_mha(4, 2);

    let mut mha = builder();
    let x = Node::new(x_data.clone(), shape.clone());
    let out = mha.forward_batch(x.clone());

    let grad_inj: Vec<f32> = vec![1.0; out.borrow().data.len()];
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let analytic = x.borrow().grad.clone();
    let numeric = numerical_grad_x(&builder, &x_data, &shape);

    println!("X analytic: {analytic:?}");
    println!("X numeric:  {numeric:?}");

    for i in 0..x_data.len() {
        let diff = (analytic[i] - numeric[i]).abs();
        assert!(diff < 2e-2, "X grad mismatch at {}: {} vs {}", i, analytic[i], numeric[i]);
    }
    println!("multihead batch input gradcheck ok");
}