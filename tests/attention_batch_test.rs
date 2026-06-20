use neuralrs::nn::attention::attention_batch;
use neuralrs::autograd::node::{Node, backward_graph};

fn numerical_grad(
    target: char,
    q: &Vec<f32>, k: &Vec<f32>, v: &Vec<f32>, shape: &Vec<usize>,
) -> Vec<f32> {
    let h = 1e-3;
    let len = match target { 'q' => q.len(), 'k' => k.len(), _ => v.len() };
    let mut grad = vec![0.0; len];

    for i in 0..len {
        let mut qp = q.clone(); let mut kp = k.clone(); let mut vp = v.clone();
        match target { 'q' => qp[i] += h, 'k' => kp[i] += h, _ => vp[i] += h };
        let out_p = attention_batch(
            Node::new(qp, shape.clone()),
            Node::new(kp, shape.clone()),
            Node::new(vp, shape.clone()),
        );
        let loss_p: f32 = out_p.borrow().data.iter().sum();

        let mut qm = q.clone(); let mut km = k.clone(); let mut vm = v.clone();
        match target { 'q' => qm[i] -= h, 'k' => km[i] -= h, _ => vm[i] -= h };
        let out_m = attention_batch(
            Node::new(qm, shape.clone()),
            Node::new(km, shape.clone()),
            Node::new(vm, shape.clone()),
        );
        let loss_m: f32 = out_m.borrow().data.iter().sum();

        grad[i] = (loss_p - loss_m) / (2.0 * h);
    }
    grad
}

#[test]
fn attention_batch_shape() {
    let shape = vec![2, 3, 4];
    let n = 2 * 3 * 4;
    let q = Node::new((0..n).map(|i| (i as f32) * 0.01).collect(), shape.clone());
    let k = Node::new((0..n).map(|i| (i as f32) * 0.02).collect(), shape.clone());
    let v = Node::new((0..n).map(|i| (i as f32) * 0.03).collect(), shape.clone());

    let out = attention_batch(q, k, v);
    assert_eq!(out.borrow().shape, vec![2, 3, 4]);
    assert!(out.borrow().data.iter().all(|x| x.is_finite()));
    println!("attention batch shape ok");
}

#[test]
fn attention_batch_gradcheck() {
    let shape = vec![2, 2, 2];
    let q = vec![0.1, 0.2, 0.3, 0.4,  0.2, 0.1, 0.0, 0.3];
    let k = vec![0.2, 0.1, 0.4, 0.3,  0.1, 0.2, 0.3, 0.0];
    let v = vec![0.5, 0.6, 0.7, 0.8,  0.4, 0.3, 0.2, 0.1];

    let qn = Node::new(q.clone(), shape.clone());
    let kn = Node::new(k.clone(), shape.clone());
    let vn = Node::new(v.clone(), shape.clone());

    let out = attention_batch(qn.clone(), kn.clone(), vn.clone());
    let grad_inj: Vec<f32> = vec![1.0; out.borrow().data.len()];
    out.borrow_mut().grad = grad_inj;
    backward_graph(&out);

    let q_analytic = qn.borrow().grad.clone();
    let k_analytic = kn.borrow().grad.clone();
    let v_analytic = vn.borrow().grad.clone();

    let q_numeric = numerical_grad('q', &q, &k, &v, &shape);
    let k_numeric = numerical_grad('k', &q, &k, &v, &shape);
    let v_numeric = numerical_grad('v', &q, &k, &v, &shape);

    println!("Q analytic: {q_analytic:?}");
    println!("Q numeric:  {q_numeric:?}");
    println!("K analytic: {k_analytic:?}");
    println!("K numeric:  {k_numeric:?}");
    println!("V analytic: {v_analytic:?}");
    println!("V numeric:  {v_numeric:?}");

    for i in 0..q.len() {
        assert!((q_analytic[i] - q_numeric[i]).abs() < 2e-2, "Q mismatch at {}: {} vs {}", i, q_analytic[i], q_numeric[i]);
    }
    for i in 0..k.len() {
        assert!((k_analytic[i] - k_numeric[i]).abs() < 2e-2, "K mismatch at {}: {} vs {}", i, k_analytic[i], k_numeric[i]);
    }
    for i in 0..v.len() {
        assert!((v_analytic[i] - v_numeric[i]).abs() < 2e-2, "V mismatch at {}: {} vs {}", i, v_analytic[i], v_numeric[i]);
    }

    println!("attention batch gradcheck ok");
}