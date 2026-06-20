use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::linear::Linear;
use neuralrs::nn::sequential::Sequential;
use neuralrs::ops::elementwise::mse_node::mse_node;
use neuralrs::optim::sgd::SGD;
use neuralrs::nn::activations::relu::ReLU;
use neuralrs::autograd::node::Node;
use neuralrs::init::he;

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
    let mut sgd = SGD { lr: 0.001, momentum: 0.5, velocity: vec![] };

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
        sgd.step(&mut model.list);

        println!("Epoch {epoch}: loss = {loss}");
    }
}