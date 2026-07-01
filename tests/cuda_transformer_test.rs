#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::{Linear as GpuLinear, LayerNorm as GpuLN, MultiHeadAttention as GpuMHA, TransformerBlock};
use neuralrs::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < tol, "{what} [{i}]: gpu {} cpu {}", a[i], b[i]);
    }
}

#[test]
fn cuda_transformer_block() {
    let (batch, seq, d, heads, d_ff) = (2usize, 3usize, 6usize, 2usize, 12usize);
    let eps = 1e-5f32;
    let mk = |len: usize, s: usize| -> Vec<f32> { (0..len).map(|i| ((i + s) % 13) as f32 * 0.1 - 0.6).collect() };

    let (wq, wk, wv, wo) = (mk(d * d, 1), mk(d * d, 2), mk(d * d, 3), mk(d * d, 4));
    let (g1, b1, g2, b2) = (mk(d, 5), mk(d, 6), mk(d, 7), mk(d, 8));
    let (f1w, f1b, f2w, f2b) = (mk(d * d_ff, 9), mk(d_ff, 10), mk(d_ff * d, 11), mk(d, 12));
    let x = mk(batch * seq * d, 20);
    let seed = mk(batch * seq * d, 30);

    let t = |data: &Vec<f32>, sh: Vec<usize>| Tensor::new(data.clone(), sh);
    let ln = |g: &Vec<f32>, b: &Vec<f32>| neuralrs::nn::normalization::LayerNorm {
        gamma: Tensor::new(g.clone(), vec![d]), beta: Tensor::new(b.clone(), vec![d]),
        epsilon: eps, num_features: d,
        gamma_grad: Rc::new(RefCell::new(vec![0.0; d])), beta_grad: Rc::new(RefCell::new(vec![0.0; d])),
    };
    let mut tb = neuralrs::nn::transformer_block::TransformerBlock {
        mha: neuralrs::nn::multihead::MultiHeadAttention {
            w_q: t(&wq, vec![d, d]), w_k: t(&wk, vec![d, d]), w_v: t(&wv, vec![d, d]), w_o: t(&wo, vec![d, d]),
            d_model: d, num_heads: heads,
            w_q_node: None, w_k_node: None, w_v_node: None, w_o_node: None,
        },
        norm1: ln(&g1, &b1),
        norm2: ln(&g2, &b2),
        ff1: neuralrs::nn::linear::Linear { weights: t(&f1w, vec![d, d_ff]), bias: t(&f1b, vec![d_ff]), weights_node: None, bias_node: None },
        ff2: neuralrs::nn::linear::Linear { weights: t(&f2w, vec![d_ff, d]), bias: t(&f2b, vec![d]), weights_node: None, bias_node: None },
        d_model: d, d_ff,
    };
    let cx = Node::new(x.clone(), vec![batch, seq, d]);
    let cout = tb.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    tb.sync_grads();
    let cg = [
        tb.mha.w_q.grad.clone(), tb.mha.w_k.grad.clone(), tb.mha.w_v.grad.clone(), tb.mha.w_o.grad.clone(),
        tb.norm1.gamma.grad.clone(), tb.norm1.beta.grad.clone(), tb.norm2.gamma.grad.clone(), tb.norm2.beta.grad.clone(),
        tb.ff1.weights.grad.clone(), tb.ff1.bias.grad.clone(), tb.ff2.weights.grad.clone(), tb.ff2.bias.grad.clone(),
    ];
    let cdx = cx.borrow().grad.clone();

    let node = |data: &Vec<f32>, sh: Vec<usize>| { let n = Node::new(data.clone(), sh); gpu::to_cuda(&n); n };
    let (gwq, gwk, gwv, gwo) = (node(&wq, vec![d, d]), node(&wk, vec![d, d]), node(&wv, vec![d, d]), node(&wo, vec![d, d]));
    let (gg1, gb1, gg2, gb2) = (node(&g1, vec![d]), node(&b1, vec![d]), node(&g2, vec![d]), node(&b2, vec![d]));
    let (gf1w, gf1b, gf2w, gf2b) = (node(&f1w, vec![d, d_ff]), node(&f1b, vec![d_ff]), node(&f2w, vec![d_ff, d]), node(&f2b, vec![d]));
    let gtb = TransformerBlock {
        mha: GpuMHA::new(gwq.clone(), gwk.clone(), gwv.clone(), gwo.clone(), d, heads),
        norm1: GpuLN::new(gg1.clone(), gb1.clone(), eps),
        norm2: GpuLN::new(gg2.clone(), gb2.clone(), eps),
        ff1: GpuLinear::new(gf1w.clone(), gf1b.clone()),
        ff2: GpuLinear::new(gf2w.clone(), gf2b.clone()),
        d_model: d, d_ff,
    };
    let gx = node(&x, vec![batch, seq, d]);
    let gout = gtb.forward(&gx);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gg = [
        gpu::read_grad(&gwq), gpu::read_grad(&gwk), gpu::read_grad(&gwv), gpu::read_grad(&gwo),
        gpu::read_grad(&gg1), gpu::read_grad(&gb1), gpu::read_grad(&gg2), gpu::read_grad(&gb2),
        gpu::read_grad(&gf1w), gpu::read_grad(&gf1b), gpu::read_grad(&gf2w), gpu::read_grad(&gf2b),
    ];
    let gdx = gpu::read_grad(&gx);

    close(&gf, &cf, 3e-3, "transformer fwd");
    let names = ["w_q", "w_k", "w_v", "w_o", "norm1.gamma", "norm1.beta", "norm2.gamma", "norm2.beta", "ff1.w", "ff1.b", "ff2.w", "ff2.b"];
    for j in 0..12 { close(&gg[j], &cg[j], 3e-3, &format!("transformer d{}", names[j])); }
    close(&gdx, &cdx, 3e-3, "transformer dx");
    println!("transformer_block: gpu matches cpu ({} params + input all gradient-checked)", names.len());
}