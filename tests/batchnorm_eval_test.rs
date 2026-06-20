use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::batchnorm::BatchNorm;
use neuralrs::autograd::node::Node;

fn make_bn() -> BatchNorm {
    BatchNorm {
        gamma: Tensor::new(vec![1.0, 1.0], vec![2]),
        beta: Tensor::new(vec![0.0, 0.0], vec![2]),
        epsilon: 1e-5,
        num_features: 2,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        running_mean: vec![0.0; 2],
        running_var: vec![1.0; 2],
        momentum: 0.9,
        training: true,
    }
}

#[test]
fn batchnorm_updates_running_stats() {
    let mut bn = make_bn();

    let input = Node::new(vec![2.0, 10.0, 4.0, 20.0, 6.0, 30.0], vec![3, 2]);
    let _ = bn.forward(input);

    println!("running_mean after train: {:?}", bn.running_mean);
    println!("running_var after train: {:?}", bn.running_var);

    assert!((bn.running_mean[0] - 0.4).abs() < 1e-4);
    assert!((bn.running_mean[1] - 2.0).abs() < 1e-4);

    println!("running stats update ok");
}

#[test]
fn batchnorm_eval_single_sample() {
    let mut bn = make_bn();

    for _ in 0..5 {
        let input = Node::new(vec![2.0, 10.0, 4.0, 20.0, 6.0, 30.0], vec![3, 2]);
        let _ = bn.forward(input);
    }

    bn.set_training(false);
    let single = Node::new(vec![4.0, 20.0], vec![1, 2]);
    let out = bn.forward(single);

    let data = out.borrow().data.clone();
    println!("eval output (batch=1): {data:?}");

    assert!(data.iter().all(|x| x.is_finite()), "eval on batch=1 produced non-finite values!");
    assert_eq!(out.borrow().shape, vec![1, 2]);

    println!("batchnorm eval single sample ok");
}