use neuralrs::nn::normalization::LayerNorm;
use neuralrs::nn::module::Module;
use neuralrs::autograd::node::Node;
use neuralrs::tensor::Tensor;

fn numerical_grad(layer: &mut LayerNorm, input_data: &Vec<f32>, shape: &Vec<usize>) -> Vec<f32> {
    let h = 1e-3;
    let mut grad = vec![0.0; input_data.len()];

    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += h;
        let out_plus = layer.forward(Node::new(plus, shape.clone()));
        let loss_plus: f32 = out_plus.borrow().data.iter().map(|x| x * x).sum();

        let mut minus = input_data.clone();
        minus[i] -= h;
        let out_minus = layer.forward(Node::new(minus, shape.clone()));
        let loss_minus: f32 = out_minus.borrow().data.iter().map(|x| x * x).sum();

        grad[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    grad
}

#[test]
fn gradcheck_layernorm() {
    let make_layer = || LayerNorm {
        gamma: Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        beta: Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        epsilon: 1e-5,
        num_features: 3,
        gamma_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 3])),
        beta_grad: std::rc::Rc::new(std::cell::RefCell::new(vec![0.0; 3])),
    };

    let input_data = vec![2.0, 5.0, 1.0];
    let shape = vec![1, 3];

    let mut layer_n = make_layer();
    let numeric = numerical_grad(&mut layer_n, &input_data, &shape);

    let mut layer_a = make_layer();
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer_a.forward(input.clone());

    let grad_inj: Vec<f32> = output.borrow().data.iter().map(|x| 2.0 * x).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();

    let analytic = input.borrow().grad.clone();

    println!("numeric:  {numeric:?}");
    println!("analytic: {analytic:?}");

    for i in 0..numeric.len() {
        let diff = (numeric[i] - analytic[i]).abs();
        println!("diff[{i}] = {diff}");
        assert!(diff < 1e-2, "gradient mismatch at {i}");
    }
}