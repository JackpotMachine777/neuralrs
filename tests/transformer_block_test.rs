use neuralrs::nn::transformer_block::TransformerBlock;
use neuralrs::nn::multihead::MultiHeadAttention;
use neuralrs::nn::normalization::LayerNorm;
use neuralrs::nn::linear::Linear;
use neuralrs::autograd::node::{Node, backward_graph};
use neuralrs::tensor::Tensor;

fn make_block(d_model: usize, d_ff: usize, num_heads: usize) -> TransformerBlock {
    let mk = |rows: usize, cols: usize, seed: f32| -> Vec<f32> {
        (0..rows*cols).map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.1).collect()
    };
    let ln = |features: usize| LayerNorm {
        gamma: Tensor::new(vec![1.0; features], vec![features]),
        beta: Tensor::new(vec![0.0; features], vec![features]),
        epsilon: 1e-5,
        num_features: features,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; features])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; features])),
    };
    TransformerBlock {
        mha: MultiHeadAttention {
            w_q: Tensor::new(mk(d_model, d_model, 1.0), vec![d_model, d_model]),
            w_k: Tensor::new(mk(d_model, d_model, 2.0), vec![d_model, d_model]),
            w_v: Tensor::new(mk(d_model, d_model, 3.0), vec![d_model, d_model]),
            w_o: Tensor::new(mk(d_model, d_model, 4.0), vec![d_model, d_model]),
            d_model,
            num_heads,
            w_q_node: None, w_k_node: None, w_v_node: None, w_o_node: None,
        },
        norm1: ln(d_model),
        norm2: ln(d_model),
        ff1: Linear {
            weights: Tensor::new(mk(d_model, d_ff, 5.0), vec![d_model, d_ff]),
            bias: Tensor::new(vec![0.0; d_ff], vec![d_ff]),
            weights_node: None, bias_node: None,
        },
        ff2: Linear {
            weights: Tensor::new(mk(d_ff, d_model, 6.0), vec![d_ff, d_model]),
            bias: Tensor::new(vec![0.0; d_model], vec![d_model]),
            weights_node: None, bias_node: None,
        },
        d_model,
        d_ff,
    }
}

fn loss_of(out: &std::rc::Rc<std::cell::RefCell<Node>>) -> f32 {
    out.borrow().data.iter().enumerate().map(|(j, x)| (j as f32 + 1.0) * x).sum()
}

#[test]
fn transformer_block_shape() {
    let mut block = make_block(4, 8, 2);
    let n = 2 * 3 * 4;
    let x = Node::new((0..n).map(|i| (i as f32) * 0.01).collect(), vec![2, 3, 4]);
    let out = block.forward(x);

    assert_eq!(out.borrow().shape, vec![2, 3, 4]);
    assert!(out.borrow().data.iter().all(|v| v.is_finite()));
    println!("transformer block shape ok");
}

fn numeric_grad_at(x_data: &Vec<f32>, shape: &Vec<usize>, i: usize) -> f32 {
    let h = 1e-3;

    let loss_p = {
        let mut xp = x_data.clone();
        xp[i] += h;
        let mut b = make_block(4, 8, 2);
        let out = b.forward_no_final_norm(Node::new(xp, shape.clone()));
        loss_of(&out)
    };

    let loss_m = {
        let mut xm = x_data.clone();
        xm[i] -= h;
        let mut b = make_block(4, 8, 2);
        let out = b.forward_no_final_norm(Node::new(xm, shape.clone()));
        loss_of(&out)
    };

    (loss_p - loss_m) / (2.0 * h)
}

#[test]
fn transformer_block_gradcheck_input() {
    let shape = vec![2, 2, 4];
    let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.03).collect();

    let analytic = {
        let mut block = make_block(4, 8, 2);
        let x = Node::new(x_data.clone(), shape.clone());
        let out = block.forward_no_final_norm(x.clone());
        let grad_inj: Vec<f32> = (0..out.borrow().data.len()).map(|j| j as f32 + 1.0).collect();
        out.borrow_mut().grad = grad_inj;
        backward_graph(&out);
        
        x.borrow().grad.clone()
    };

    let mut numeric = vec![0.0; x_data.len()];
    for i in 0..x_data.len() {
        numeric[i] = numeric_grad_at(&x_data, &shape, i);
    }

    println!("X analytic: {analytic:?}");
    println!("X numeric:  {numeric:?}");

    for i in 0..x_data.len() {
        let diff = (analytic[i] - numeric[i]).abs();
        assert!(diff < 3e-2, "X grad mismatch at {}: {} vs {}", i, analytic[i], numeric[i]);
    }
    println!("transformer block gradcheck (no final norm) ok");
}