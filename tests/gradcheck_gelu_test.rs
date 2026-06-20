use neuralrs::nn::module::Module;
use neuralrs::nn::activations::gelu::GELU;
use neuralrs::autograd::node::Node;

fn loss_of(out: &Vec<f32>) -> f32 {
    out.iter().enumerate().map(|(i, x)| (i as f32 + 1.0) * x).sum()
}

#[test]
fn gradcheck_gelu() {
    let input_data = vec![-2.0, -0.5, 0.5, 1.0, 3.0];
    let shape = vec![1, 5];

    let h = 1e-3;
    let mut numeric = vec![0.0; input_data.len()];
    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += h;
        let mut lp = GELU {};
        let op = lp.forward(Node::new(plus, shape.clone()));
        let loss_plus = loss_of(&op.borrow().data);

        let mut minus = input_data.clone();
        minus[i] -= h;
        let mut lm = GELU {};
        let om = lm.forward(Node::new(minus, shape.clone()));
        let loss_minus = loss_of(&om.borrow().data);

        numeric[i] = (loss_plus - loss_minus) / (2.0 * h);
    }

    let mut layer = GELU {};
    let input = Node::new(input_data.clone(), shape.clone());
    let output = layer.forward(input.clone());
    let out_len = output.borrow().data.len();
    let grad_inj: Vec<f32> = (0..out_len).map(|i| i as f32 + 1.0).collect();
    output.borrow_mut().grad = grad_inj;
    output.borrow_mut().backward();
    let analytic = input.borrow().grad.clone();

    println!("numeric:  {numeric:?}");
    println!("analytic: {analytic:?}");

    for i in 0..numeric.len() {
        let diff = (numeric[i] - analytic[i]).abs();
        assert!(diff < 1e-2, "gelu grad mismatch at {}: {} vs {}", i, numeric[i], analytic[i]);
    }
    println!("gelu gradcheck ok");
}