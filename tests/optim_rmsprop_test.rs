// adam_test.rs
use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::linear::Linear;
use rstorch::nn::sequential::Sequential;
use rstorch::ops::elementwise::mse_node::mse_node;
use rstorch::optim::rmsprop::RMSProp;
use rstorch::nn::activations::relu::ReLU;
use rstorch::autograd::node::Node;
use rstorch::init::he;

#[test]
fn prototype_model_test(){
    let mut model = Sequential{
        list: vec![
            Box::new(Linear{
                weights: Tensor::new(he::he(2, 2), vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                weights_node: None,
                bias_node: None,
            }),
            Box::new(ReLU {}),
            Box::new(Linear{
                weights: Tensor::new(he::he(2, 2), vec![2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                weights_node: None,
                bias_node: None,
            }),
        ],
    };

    let input = Node::new(vec![1.0, 2.0], vec![1, 2]);
    let target = Node::new(vec![0.0, 1.0], vec![1, 2]);
    let mut rms = RMSProp { lr: 0.01, beta: 0.9, epsilon: 1e-8, v: vec![] };

    for epoch in 0..100 {
        model.zero_grad();
        let output = model.forward(input.clone());
        let loss = mse_node(output.clone(), target.clone());

        let grad_data: Vec<f32> = {
            let p = output.borrow();
            let t = target.borrow();
            let n = p.data.len();
            (0..n).map(|i| 2.0 * (p.data[i] - t.data[i]) / n as f32).collect()
        };
        output.borrow_mut().grad = grad_data;
        output.borrow_mut().backward();

        model.sync_grads();
        rms.step(&mut model.list);

        println!("Epoch {}: loss = {}", epoch, loss);
    }
}