use neuralrs::nn::module::Module;
use neuralrs::nn::linear::Linear;
use neuralrs::tensor::Tensor;
use neuralrs::autograd::node::Node;
use neuralrs::optim::adagrad::Adagrad;
use neuralrs::optim::nesterov::NesterovSGD;
use neuralrs::optim::nadam::NAdam;

fn make_linear(init: f32) -> Linear {
    Linear {
        weights: Tensor::new(vec![init], vec![1, 1]),
        bias: Tensor::new(vec![0.0], vec![1]),
        weights_node: None,
        bias_node: None,
    }
}

fn run<F: FnMut(&mut Vec<Box<dyn Module>>)>(mut step_fn: F, mut model: Vec<Box<dyn Module>>, steps: usize) -> f32 {
    let target = 3.0;
    for _ in 0..steps {
        let input = Node::new(vec![1.0], vec![1, 1]);
        for m in model.iter_mut() { m.zero_grad(); }

        let out = model[0].forward(input);
        let out_val = out.borrow().data[0];
        let dloss = 2.0 * (out_val - target);
        out.borrow_mut().grad = vec![dloss];
        out.borrow_mut().backward();

        for m in model.iter_mut() { m.sync_grads(); }

        step_fn(&mut model);
    }
    let params = model[0].parameters();
    let w = params[0].storage.data[0];
    let b = params[1].storage.data[0];
    w + b
}

#[test]
fn adagrad_converges() {
    let mut opt = Adagrad { lr: 0.5, epsilon: 1e-8, g_sum: Vec::new(), t: 0 };
    let model: Vec<Box<dyn Module>> = vec![Box::new(make_linear(0.0))];
    let final_w = run(|m| opt.step(m), model, 200);
    println!("adagrad final w: {final_w} (target 3.0)");
    assert!((final_w - 3.0).abs() < 0.1, "adagrad didn't converge: {final_w}");
    println!("adagrad converges ok");
}

#[test]
fn nesterov_converges() {
    let mut opt = NesterovSGD { lr: 0.05, momentum: 0.9, velocity: Vec::new(), t: 0 };
    let model: Vec<Box<dyn Module>> = vec![Box::new(make_linear(0.0))];
    let final_w = run(|m| opt.step(m), model, 200);
    println!("nesterov final w: {final_w} (target 3.0)");
    assert!((final_w - 3.0).abs() < 0.1, "nesterov didn't converge: {final_w}");
    println!("nesterov converges ok");
}

#[test]
fn nadam_converges() {
    let mut opt = NAdam { lr: 0.1, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, t: 0, m: Vec::new(), v: Vec::new() };
    let model: Vec<Box<dyn Module>> = vec![Box::new(make_linear(0.0))];
    let final_w = run(|m| opt.step(m), model, 200);
    println!("nadam final w: {final_w} (target 3.0)");
    assert!((final_w - 3.0).abs() < 0.1, "nadam didn't converge: {final_w}");
    println!("nadam converges ok");
}