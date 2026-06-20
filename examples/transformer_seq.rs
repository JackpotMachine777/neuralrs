use neuralrs::tensor::Tensor;
use neuralrs::nn::linear::Linear;
use neuralrs::nn::module::Module;
use neuralrs::nn::embedding::Embedding;
use neuralrs::nn::positional::PositionalEncoding;
use neuralrs::nn::transformer_block::TransformerBlock;
use neuralrs::nn::multihead::MultiHeadAttention;
use neuralrs::nn::normalization::LayerNorm;
use neuralrs::nn::loss::{Loss, CrossEntropyLoss};
use neuralrs::autograd::node::{Node, backward_graph};
use neuralrs::autograd::graph;
use neuralrs::init::he;
use neuralrs::optim::adamw::ADAMW;

use std::rc::Rc;
use std::cell::RefCell;

const VOCAB: usize = 10;
const SEQ: usize = 6;
const D_MODEL: usize = 32;
const D_FF: usize = 64;
const HEADS: usize = 4;

fn make_linear(fan_in: usize, fan_out: usize) -> Linear {
    Linear {
        weights: Tensor::new(he::he(fan_in, fan_out), vec![fan_in, fan_out]),
        bias: Tensor::new(vec![0.0; fan_out], vec![fan_out]),
        weights_node: None,
        bias_node: None,
    }
}

fn make_ln(features: usize) -> LayerNorm {
    LayerNorm {
        gamma: Tensor::new(vec![1.0; features], vec![features]),
        beta: Tensor::new(vec![0.0; features], vec![features]),
        epsilon: 1e-5,
        num_features: features,
        gamma_grad: Rc::new(RefCell::new(vec![0.0; features])),
        beta_grad: Rc::new(RefCell::new(vec![0.0; features])),
    }
}

fn make_block() -> TransformerBlock {
    TransformerBlock {
        mha: MultiHeadAttention {
            w_q: Tensor::new(he::he(D_MODEL, D_MODEL), vec![D_MODEL, D_MODEL]),
            w_k: Tensor::new(he::he(D_MODEL, D_MODEL), vec![D_MODEL, D_MODEL]),
            w_v: Tensor::new(he::he(D_MODEL, D_MODEL), vec![D_MODEL, D_MODEL]),
            w_o: Tensor::new(he::he(D_MODEL, D_MODEL), vec![D_MODEL, D_MODEL]),
            d_model: D_MODEL,
            num_heads: HEADS,
            w_q_node: None, w_k_node: None, w_v_node: None, w_o_node: None,
        },
        norm1: make_ln(D_MODEL),
        norm2: make_ln(D_MODEL),
        ff1: make_linear(D_MODEL, D_FF),
        ff2: make_linear(D_FF, D_MODEL),
        d_model: D_MODEL,
        d_ff: D_FF,
    }
}

fn make_sample() -> (Vec<usize>, usize) {
    let seq: Vec<usize> = (0..SEQ)
        .map(|_| (rand::random::<f32>() * VOCAB as f32) as usize % VOCAB)
        .collect();
    let sorted = seq.windows(2).all(|w| w[0] <= w[1]);
    (seq, if sorted { 1 } else { 0 })
}

fn mean_pool(x: Rc<RefCell<Node>>, seq: usize, d: usize) -> Rc<RefCell<Node>> {
    let x2 = graph::reshape(x, vec![seq, d]);
    let ones = Node::new(vec![1.0 / seq as f32; seq], vec![1, seq]);
    graph::matmul(ones, x2)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = v[0];
    for i in 1..v.len() {
        if v[i] > bv { bv = v[i]; best = i; }
    }
    best
}

fn main() {
    println!("Transformer sequence classification: is the sequence sorted?");
    println!("vocab={VOCAB} seq_len={SEQ} d_model={D_MODEL} heads={HEADS} d_ff={D_FF}");

    let mut embedding = Embedding {
        weight: Tensor::new(he::he(VOCAB, D_MODEL), vec![VOCAB, D_MODEL]),
        vocab_size: VOCAB,
        embedding_dim: D_MODEL,
        weight_node: None,
    };
    let pos_enc = PositionalEncoding::new(D_MODEL, SEQ);
    let mut block = make_block();
    let mut classifier = make_linear(D_MODEL, 2);

    let loss_fn = CrossEntropyLoss;

    let mut opt_emb = ADAMW { lr: 0.0015, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0, t: 0, m: Vec::new(), v: Vec::new() };
    let mut opt_block = ADAMW { lr: 0.0015, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0, t: 0, m: Vec::new(), v: Vec::new() };
    let mut opt_cls = ADAMW { lr: 0.0015, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0, t: 0, m: Vec::new(), v: Vec::new() };

    let steps = 6000;
    let report_every = 500;
    let mut running_loss = 0.0;
    let mut correct = 0;

    for step in 0..steps {
        let (seq, label) = make_sample();

        embedding.zero_grad();
        block.zero_grad();
        classifier.zero_grad();

        let emb = embedding.forward(&seq);
        let emb = pos_enc.forward(emb);
        let x = graph::reshape(emb, vec![1, SEQ, D_MODEL]);
        let h = block.forward(x);
        let pooled = mean_pool(h, SEQ, D_MODEL);
        let logits = classifier.forward(pooled);

        let mut tgt = vec![0.0; 2];
        tgt[label] = 1.0;
        let target = Node::new(tgt, vec![1, 2]);

        let loss = loss_fn.forward(&logits, &target);
        running_loss += loss;
        if argmax(&logits.borrow().data) == label {
            correct += 1;
        }

        loss_fn.backward(&logits, &target);
        backward_graph(&logits);

        embedding.sync_grads();
        block.sync_grads();
        classifier.sync_grads();

        opt_emb.step_params(&mut embedding.parameters());
        opt_block.step_params(&mut block.parameters());
        opt_cls.step_params(&mut classifier.parameters());

        if (step + 1) % report_every == 0 {
            println!(
                "step {:>5}: avg loss {:.4}, train acc {:.1}%",
                step + 1,
                running_loss / report_every as f32,
                correct as f32 / report_every as f32 * 100.0
            );
            running_loss = 0.0;
            correct = 0;
        }
    }

    let eval_n = 2000;
    let mut eval_correct = 0;
    for _ in 0..eval_n {
        let (seq, label) = make_sample();
        embedding.weight_node = None;
        let emb = embedding.forward(&seq);
        let emb = pos_enc.forward(emb);
        let x = graph::reshape(emb, vec![1, SEQ, D_MODEL]);
        let h = block.forward(x);
        let pooled = mean_pool(h, SEQ, D_MODEL);
        let logits = classifier.forward(pooled);
        if argmax(&logits.borrow().data) == label {
            eval_correct += 1;
        }
    }
    println!("\nFinal eval accuracy: {:.1}% ({} samples)", eval_correct as f32 / eval_n as f32 * 100.0, eval_n);
}