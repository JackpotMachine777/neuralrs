#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::{Linear as GpuLinear, MultiHeadAttention as GpuMHA, PositionalEncoding as GpuPE, SelfAttention as GpuSA};
use neuralrs::nn::module::Module;
use neuralrs::tensor::Tensor;

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < tol, "{what} [{i}]: gpu {} cpu {}", a[i], b[i]);
    }
}

#[test]
fn cuda_linear() {
    let (batch, in_d, out_d) = (3usize, 4usize, 5usize);
    let x: Vec<f32> = (0..batch * in_d).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let w: Vec<f32> = (0..in_d * out_d).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let b: Vec<f32> = (0..out_d).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
    let seed: Vec<f32> = (0..batch * out_d).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let mut lin = neuralrs::nn::linear::Linear {
        weights: Tensor::new(w.clone(), vec![in_d, out_d]),
        bias: Tensor::new(b.clone(), vec![out_d]),
        weights_node: None, bias_node: None,
    };
    let cx = Node::new(x.clone(), vec![batch, in_d]);
    let cout = lin.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    lin.sync_grads();
    let (cdw, cdb) = (lin.weights.grad.clone(), lin.bias.grad.clone());
    let cdx = cx.borrow().grad.clone();

    let gw = Node::new(w, vec![in_d, out_d]);
    let gb = Node::new(b, vec![out_d]);
    gpu::to_cuda(&gw); gpu::to_cuda(&gb);
    let glin = GpuLinear::new(gw.clone(), gb.clone());
    let gx = Node::new(x, vec![batch, in_d]);
    gpu::to_cuda(&gx);
    let gout = glin.forward(&gx);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gdw, gdb) = (gpu::read_grad(&gw), gpu::read_grad(&gb));
    let gdx = gpu::read_grad(&gx);

    close(&gf, &cf, 1e-4, "linear fwd");
    close(&gdw, &cdw, 1e-4, "linear dw");
    close(&gdb, &cdb, 1e-4, "linear db");
    close(&gdx, &cdx, 1e-4, "linear dx");
    println!("linear: gpu matches cpu");
}

#[test]
fn cuda_positional() {
    let (seq, d, max_len) = (5usize, 8usize, 16usize);
    let x: Vec<f32> = (0..seq * d).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let seed: Vec<f32> = (0..seq * d).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let cpe = neuralrs::nn::positional::PositionalEncoding::new(d, max_len);
    let cx = Node::new(x.clone(), vec![seq, d]);
    let cout = cpe.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    let cdx = cx.borrow().grad.clone();

    let gpe = GpuPE::new(d, max_len);
    let gx = Node::new(x, vec![seq, d]);
    gpu::to_cuda(&gx);
    let gout = gpe.forward(&gx);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let gdx = gpu::read_grad(&gx);

    close(&gf, &cf, 1e-5, "positional fwd");
    close(&gdx, &cdx, 1e-5, "positional dx");
    println!("positional: gpu matches cpu");
}

#[test]
fn cuda_self_attention() {
    let (seq, d_model, d_k) = (4usize, 6usize, 4usize);
    let x: Vec<f32> = (0..seq * d_model).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let wq: Vec<f32> = (0..d_model * d_k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let wk: Vec<f32> = (0..d_model * d_k).map(|i| (i % 9) as f32 * 0.1 - 0.4).collect();
    let wv: Vec<f32> = (0..d_model * d_k).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let seed: Vec<f32> = (0..seq * d_k).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let mut sa = neuralrs::nn::self_attention::SelfAttention {
        w_q: Tensor::new(wq.clone(), vec![d_model, d_k]),
        w_k: Tensor::new(wk.clone(), vec![d_model, d_k]),
        w_v: Tensor::new(wv.clone(), vec![d_model, d_k]),
        d_model, d_k,
        w_q_node: None, w_k_node: None, w_v_node: None,
    };
    let cx = Node::new(x.clone(), vec![seq, d_model]);
    let cout = sa.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    sa.sync_grads();
    let (cdq, cdk, cdv) = (sa.w_q.grad.clone(), sa.w_k.grad.clone(), sa.w_v.grad.clone());
    let cdx = cx.borrow().grad.clone();

    let gwq = Node::new(wq, vec![d_model, d_k]);
    let gwk = Node::new(wk, vec![d_model, d_k]);
    let gwv = Node::new(wv, vec![d_model, d_k]);
    gpu::to_cuda(&gwq); gpu::to_cuda(&gwk); gpu::to_cuda(&gwv);
    let gsa = GpuSA::new(gwq.clone(), gwk.clone(), gwv.clone(), d_model, d_k);
    let gx = Node::new(x, vec![seq, d_model]);
    gpu::to_cuda(&gx);
    let gout = gsa.forward(&gx);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gdq, gdk, gdv) = (gpu::read_grad(&gwq), gpu::read_grad(&gwk), gpu::read_grad(&gwv));
    let gdx = gpu::read_grad(&gx);

    close(&gf, &cf, 1e-3, "self_attn fwd");
    close(&gdq, &cdq, 1e-3, "self_attn dw_q");
    close(&gdk, &cdk, 1e-3, "self_attn dw_k");
    close(&gdv, &cdv, 1e-3, "self_attn dw_v");
    close(&gdx, &cdx, 1e-3, "self_attn dx");
    println!("self_attention: gpu matches cpu");
}

#[test]
fn cuda_multihead() {
    let (seq, d_model, num_heads) = (4usize, 6usize, 2usize);
    let x: Vec<f32> = (0..seq * d_model).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let wq: Vec<f32> = (0..d_model * d_model).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let wk: Vec<f32> = (0..d_model * d_model).map(|i| (i % 9) as f32 * 0.1 - 0.4).collect();
    let wv: Vec<f32> = (0..d_model * d_model).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let wo: Vec<f32> = (0..d_model * d_model).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
    let seed: Vec<f32> = (0..seq * d_model).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let mut mha = neuralrs::nn::multihead::MultiHeadAttention {
        w_q: Tensor::new(wq.clone(), vec![d_model, d_model]),
        w_k: Tensor::new(wk.clone(), vec![d_model, d_model]),
        w_v: Tensor::new(wv.clone(), vec![d_model, d_model]),
        w_o: Tensor::new(wo.clone(), vec![d_model, d_model]),
        d_model, num_heads,
        w_q_node: None, w_k_node: None, w_v_node: None, w_o_node: None,
    };
    let cx = Node::new(x.clone(), vec![seq, d_model]);
    let cout = mha.forward(cx.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    mha.sync_grads();
    let (cdq, cdk, cdv, cdo) = (mha.w_q.grad.clone(), mha.w_k.grad.clone(), mha.w_v.grad.clone(), mha.w_o.grad.clone());
    let cdx = cx.borrow().grad.clone();

    let gwq = Node::new(wq, vec![d_model, d_model]);
    let gwk = Node::new(wk, vec![d_model, d_model]);
    let gwv = Node::new(wv, vec![d_model, d_model]);
    let gwo = Node::new(wo, vec![d_model, d_model]);
    gpu::to_cuda(&gwq); gpu::to_cuda(&gwk); gpu::to_cuda(&gwv); gpu::to_cuda(&gwo);
    let gmha = GpuMHA::new(gwq.clone(), gwk.clone(), gwv.clone(), gwo.clone(), d_model, num_heads);
    let gx = Node::new(x, vec![seq, d_model]);
    gpu::to_cuda(&gx);
    let gout = gmha.forward(&gx);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gdq, gdk, gdv, gdo) = (gpu::read_grad(&gwq), gpu::read_grad(&gwk), gpu::read_grad(&gwv), gpu::read_grad(&gwo));
    let gdx = gpu::read_grad(&gx);

    close(&gf, &cf, 1e-3, "mha fwd");
    close(&gdq, &cdq, 1e-3, "mha dw_q");
    close(&gdk, &cdk, 1e-3, "mha dw_k");
    close(&gdv, &cdv, 1e-3, "mha dw_v");
    close(&gdo, &cdo, 1e-3, "mha dw_o");
    close(&gdx, &cdx, 1e-3, "mha dx");
    println!("multihead: gpu matches cpu");
}