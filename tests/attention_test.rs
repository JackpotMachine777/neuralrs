use rstorch::autograd::node::Node;
use rstorch::nn::attention::attention;

#[test]
fn attention_forward_backward() {
    let seq_len = 3;
    let d = 4;

    let q = Node::new((0..seq_len*d).map(|i| (i as f32 % 5.0) * 0.02 - 0.04).collect(), vec![seq_len, d]);
    let k = Node::new((0..seq_len*d).map(|i| (i as f32 % 3.0) * 0.03 - 0.03).collect(), vec![seq_len, d]);
    let v = Node::new((0..seq_len*d).map(|i| (i as f32 % 4.0) * 0.25).collect(), vec![seq_len, d]);

    let out = attention(q.clone(), k.clone(), v.clone());

    println!("out shape: {:?}", out.borrow().shape);
    assert_eq!(out.borrow().shape, vec![seq_len, d]);

    out.borrow_mut().grad = vec![1.0; seq_len * d];
    rstorch::autograd::node::backward_graph(&out);

    let q_grad: f32 = q.borrow().grad.iter().map(|x| x.abs()).sum();
    let k_grad: f32 = k.borrow().grad.iter().map(|x| x.abs()).sum();
    let v_grad: f32 = v.borrow().grad.iter().map(|x| x.abs()).sum();

    println!("q_grad: {}, k_grad: {}, v_grad: {}", q_grad, k_grad, v_grad);
    assert!(q_grad > 0.0, "Q gradient zero!");
    assert!(k_grad > 0.0, "K gradient zero!");
    assert!(v_grad > 0.0, "V gradient zero!");

    println!("attention forward/backward ok");
}

#[test]
fn attention_uniform_when_keys_equal() {
    let seq_len = 3;
    let d = 2;

    let q = Node::new(vec![0.5; seq_len * d], vec![seq_len, d]);
    let k = Node::new(vec![1.0; seq_len * d], vec![seq_len, d]);
    let v = Node::new(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![seq_len, d]);

    let out = attention(q, k, v);
    let data = out.borrow().data.clone();
    println!("out: {:?}", data);

    for row in 0..seq_len {
        assert!((data[row*d] - 2.0).abs() < 1e-4, "wiersz {} nie jest srednia", row);
        assert!((data[row*d+1] - 2.0).abs() < 1e-4);
    }

    println!("attention uniform ok");
}