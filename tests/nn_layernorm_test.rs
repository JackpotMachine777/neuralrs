use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::normalization::LayerNorm;
use neuralrs::autograd::node::Node;

#[test]
fn layernorm_test(){
    let mut layer = LayerNorm {
        gamma: Tensor::new(vec![1.0, 1.0], vec![2]),
        beta: Tensor::new(vec![0.0, 0.0], vec![2]),
        epsilon: 1e-5,
        num_features: 2,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 2])),
    };

    let input = Node::new(vec![1.0, 2.0, 10.0, 20.0], vec![2, 2]);
    let output = layer.forward(input.clone());

    println!("input:  {:?}", input.borrow().data);
    println!("output: {:?}", output.borrow().data);

    assert_eq!(output.borrow().shape, vec![2, 2]);

    let row1_mean: f32 = output.borrow().data[0..2].iter().sum::<f32>() / 2.0;
    let row2_mean: f32 = output.borrow().data[2..4].iter().sum::<f32>() / 2.0;

    println!("row1 mean: {row1_mean}");
    println!("row2 mean: {row2_mean}");

    assert!(row1_mean.abs() < 1e-5);
    assert!(row2_mean.abs() < 1e-5);
}