use neuralrs::tensor::Tensor;
use neuralrs::nn::embedding::Embedding;
use neuralrs::autograd::node::backward_graph;

fn make_embedding() -> Embedding {
    let weight = Tensor::new(vec![
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0,
    ], vec![4, 3]);
    Embedding {
        weight,
        vocab_size: 4,
        embedding_dim: 3,
        weight_node: None,
    }
}

#[test]
fn embedding_lookup() {
    let mut emb = make_embedding();
    emb.zero_grad();

    let out = emb.forward(&[2, 0, 3]);

    println!("output: {:?}", out.borrow().data);
    assert_eq!(out.borrow().shape, vec![3, 3]);
    assert_eq!(out.borrow().data, vec![
        2.0, 2.0, 2.0,
        0.0, 0.0, 0.0,
        3.0, 3.0, 3.0,
    ]);

    println!("embedding lookup ok");
}

#[test]
fn embedding_gradient_to_used_rows() {
    let mut emb = make_embedding();
    emb.zero_grad();

    let out = emb.forward(&[1, 1, 2]);

    out.borrow_mut().grad = vec![1.0; 9];
    backward_graph(&out);

    emb.sync_grads();
    let g = &emb.weight.grad;
    println!("weight grad: {g:?}");

    assert_eq!(&g[0..3], &[0.0, 0.0, 0.0]);
    assert_eq!(&g[3..6], &[2.0, 2.0, 2.0]);
    assert_eq!(&g[6..9], &[1.0, 1.0, 1.0]);
    assert_eq!(&g[9..12], &[0.0, 0.0, 0.0]);

    println!("embedding gradient ok");
}