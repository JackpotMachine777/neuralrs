use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::conv::Conv2d;
use neuralrs::nn::maxpool::MaxPool2d;
use neuralrs::nn::flatten::Flatten;
use neuralrs::nn::linear::Linear;
use neuralrs::nn::activations::relu::ReLU;
use neuralrs::autograd::node::Node;
use neuralrs::init::he;
use std::rc::Rc;
use std::cell::RefCell;

#[test]
fn cnn_end_to_end() {
    let mut conv = Conv2d {
        weight: Tensor::new(he::he(8, 1), vec![2, 1, 2, 2]),
        bias: Tensor::new(vec![0.0, 0.0], vec![2]),
        c_in: 1, c_out: 2, kh: 2, kw: 2, stride: 1,
        in_h: 4, in_w: 4,
        weight_grad: Rc::new(RefCell::new(vec![0.0; 8])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; 2])),
        padding: 0,
    };
    let mut relu = ReLU {};
    let mut pool = MaxPool2d { kernel: 2, stride: 1, channels: 2, in_h: 3, in_w: 3 };
    let mut flat = Flatten {};
    let mut fc = Linear {
        weights: Tensor::new(he::he(8, 2), vec![8, 2]),
        bias: Tensor::new(vec![0.0, 0.0], vec![2]),
        weights_node: None,
        bias_node: None,
    };

    let input_data = vec![
        1.0, 2.0, 0.0, 1.0,
        0.5, 3.0, 1.0, 0.0,
        2.0, 1.0, 0.5, 1.5,
        0.0, 1.0, 2.0, 1.0,
    ];
    let target = Node::new(vec![1.0, 0.0], vec![1, 2]);

    for epoch in 0..50 {
        conv.zero_grad();
        fc.zero_grad();

        let input = Node::new(input_data.clone(), vec![1, 4, 4]);

        let x = conv.forward(input);
        let x = relu.forward(x);
        let x = pool.forward(x);
        let x = flat.forward(x);
        let out = fc.forward(x);

        let loss: f32 = {
            let o = out.borrow();
            let t = target.borrow();
            let n = o.data.len() as f32;
            o.data.iter().zip(t.data.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / n
        };

        let grad_inj: Vec<f32> = {
            let o = out.borrow();
            let t = target.borrow();
            let n = o.data.len() as f32;
            (0..o.data.len()).map(|i| 2.0 * (o.data[i] - t.data[i]) / n).collect()
        };
        out.borrow_mut().grad = grad_inj;
        out.borrow_mut().backward();

        conv.sync_grads();
        fc.sync_grads();

        let lr = 0.01;
        for i in 0..conv.weight.storage.data.len() {
            conv.weight.storage.data[i] -= lr * conv.weight.grad[i];
        }
        for i in 0..conv.bias.storage.data.len() {
            conv.bias.storage.data[i] -= lr * conv.bias.grad[i];
        }
        for i in 0..fc.weights.storage.data.len() {
            fc.weights.storage.data[i] -= lr * fc.weights.grad[i];
        }
        for i in 0..fc.bias.storage.data.len() {
            fc.bias.storage.data[i] -= lr * fc.bias.grad[i];
        }

        println!("Epoch {epoch}: loss = {loss}");
    }
}