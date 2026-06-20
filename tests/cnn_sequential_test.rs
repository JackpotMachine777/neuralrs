use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::sequential::Sequential;
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
fn cnn_through_sequential() {
    let mut model = Sequential {
        list: vec![
            Box::new(Conv2d {
                weight: Tensor::new(he::he(8, 1), vec![2, 1, 2, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                c_in: 1, c_out: 2, kh: 2, kw: 2, stride: 1, padding: 0,
                in_h: 4, in_w: 4,
                weight_grad: Rc::new(RefCell::new(vec![0.0; 8])),
                bias_grad: Rc::new(RefCell::new(vec![0.0; 2])),
            }),
            Box::new(ReLU {}),
            Box::new(MaxPool2d { kernel: 2, stride: 1, channels: 2, in_h: 3, in_w: 3 }),
            Box::new(Flatten {}),
            Box::new(Linear {
                weights: Tensor::new(he::he(8, 2), vec![8, 2]),
                bias: Tensor::new(vec![0.0, 0.0], vec![2]),
                weights_node: None,
                bias_node: None,
            }),
        ],
    };

    let input_data = vec![
        1.0, 2.0, 0.0, 1.0,
        0.5, 3.0, 1.0, 0.0,
        2.0, 1.0, 0.5, 1.5,
        0.0, 1.0, 2.0, 1.0,
    ];
    let target = Node::new(vec![1.0, 0.0], vec![1, 2]);

    let lr = 0.01;
    let mut first_loss = 0.0;
    let mut last_loss = 0.0;

    for epoch in 0..50 {
        model.zero_grad();

        let input = Node::new(input_data.clone(), vec![1, 1, 4, 4]);
        let out = model.forward(input);

        let loss: f32 = {
            let o = out.borrow();
            let t = target.borrow();
            let n = o.data.len() as f32;
            o.data.iter().zip(t.data.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / n
        };
        if epoch == 0 { first_loss = loss; }
        last_loss = loss;

        let grad_inj: Vec<f32> = {
            let o = out.borrow();
            let t = target.borrow();
            let n = o.data.len() as f32;
            (0..o.data.len()).map(|i| 2.0 * (o.data[i] - t.data[i]) / n).collect()
        };
        out.borrow_mut().grad = grad_inj;
        out.borrow_mut().backward();

        model.sync_grads();

        for p in model.parameters() {
            for i in 0..p.storage.data.len() {
                p.storage.data[i] -= lr * p.grad[i];
            }
        }

        println!("Epoch {epoch}: loss = {loss}");
    }

    println!("first loss: {first_loss}, last loss: {last_loss}");
    assert!(last_loss < first_loss, "loss should decrease over training");
    println!("cnn through sequential ok");
}