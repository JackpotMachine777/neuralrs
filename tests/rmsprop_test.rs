use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::linear::Linear;
use rstorch::nn::sequential::Sequential;
use rstorch::ops::elementwise::{mse, mse_grad};
use rstorch::optim::rmsprop::RMSProp;
use rstorch::nn::activations::ReLU;

#[test]
fn prototype_model_test(){
    let mut model = Sequential{
        list: vec![
            Box::new(Linear{
                weights: Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                bias: Tensor::new(vec![-2.0, 0.0], vec![2]),
                last_input: None,
            }),
            Box::new(ReLU{
                last_input: None,
            }),
            Box::new(Linear {
                weights: Tensor::new(vec![1.0, 0.0, -4.0, 1.0], vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                last_input: None,
            }),
        ],
    };

    let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let target = Tensor::new(vec![0.0, 1.0], vec![1, 2]);
    let mut rmsprop = RMSProp { 
        lr: 0.01,
        beta: 0.9,
        epsilon: 1e-8,
        v: vec![],
    };

    for epoch in 0..100{
        model.zero_grad();
        let output = model.forward(&input);
        let loss = mse(&output, &target);
        let grad_loss = mse_grad(&output, &target);

        println!("Epoch: {}: loss = {}", epoch, loss);

        model.backward(&grad_loss);
        rmsprop.step(&mut model.list);
    }
}