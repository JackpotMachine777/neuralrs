use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::batchnorm::BatchNorm;
use neuralrs::autograd::node::Node;

#[test]
fn batchnorm_test(){
    let mut layer = BatchNorm {
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
    };

    let input = Node::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let output = layer.forward(input.clone());

    println!("input:  {:?}", input.borrow().data);
    println!("output: {:?}", output.borrow().data);

    assert_eq!(output.borrow().shape, vec![2, 2]);

    let mean: f32 = output.borrow().data.iter().sum::<f32>() / output.borrow().data.len() as f32;
    println!("mean: {mean}");
    assert!(mean.abs() < 1e-5);
}