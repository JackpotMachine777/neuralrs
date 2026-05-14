use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::linear::Linear;
use rstorch::nn::sequential::Sequential;
use rstorch::ops::elementwise::mse::{mse, mse_grad}
use rstorch::optim::adam::ADAM;
use rstorch::nn::activations::relu::ReLU;

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
    let mut adam = ADAM { 
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        t: 0,
        m: vec![],
        v: vec![],
    };

    for epoch in 0..100{
        model.zero_grad();
        let output = model.forward(&input);
        let loss = mse(&output, &target);
        let grad_loss = mse_grad(&output, &target);

        println!("Epoch: {}: loss = {}", epoch, loss);

        model.backward(&grad_loss);
        adam.step(&mut model.list);
    }
}