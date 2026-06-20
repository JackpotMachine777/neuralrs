use rstorch::tensor::Tensor;
use rstorch::nn::embedding::Embedding;
use rstorch::autograd::node::backward_graph;

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
fn embedding_batch_lookup() {
    let mut emb = make_embedding();
    emb.zero_grad();

    let out = emb.forward_batch(&[vec![1, 2], vec![3, 0]]);

    println!("output: {:?}", out.borrow().data);
    assert_eq!(out.borrow().shape, vec![2, 2, 3]);
    assert_eq!(out.borrow().data, vec![
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0,
        0.0, 0.0, 0.0,
    ]);

    println!("embedding batch lookup ok");
}

#[test]
fn embedding_batch_gradient_sums_across_batch() {
    let mut emb = make_embedding();
    emb.zero_grad();

    let out = emb.forward_batch(&[vec![2, 0], vec![2, 1]]);

    out.borrow_mut().grad = vec![1.0; 12];
    backward_graph(&out);

    emb.sync_grads();
    let g = &emb.weight.grad;
    println!("weight grad: {g:?}");

    assert_eq!(&g[0..3], &[1.0, 1.0, 1.0]);
    assert_eq!(&g[3..6], &[1.0, 1.0, 1.0]);
    assert_eq!(&g[6..9], &[2.0, 2.0, 2.0]);
    assert_eq!(&g[9..12], &[0.0, 0.0, 0.0]);

    println!("embedding batch gradient ok");
}