use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::sequential::Sequential;
use rstorch::nn::conv::Conv2d;
use rstorch::nn::maxpool::MaxPool2d;
use rstorch::nn::flatten::Flatten;
use rstorch::nn::linear::Linear;
use rstorch::nn::activations::relu::ReLU;
use rstorch::nn::loss::{Loss, CrossEntropyLoss};
use rstorch::data::dataloader::DataLoader;
use rstorch::data::mnist::{read_images, read_labels, read_labels_raw};
use rstorch::autograd::node::Node;
use rstorch::init::he;
use rstorch::optim::adamw::ADAMW;
use rstorch::nn::dropout::Dropout;

use std::rc::Rc;
use std::cell::RefCell;
use std::time::Instant;

fn conv(c_in: usize, c_out: usize, kh: usize, kw: usize, pad: usize, in_h: usize, in_w: usize) -> Conv2d {
    let w_len = c_out * c_in * kh * kw;
    Conv2d {
        weight: Tensor::new(he::he(c_in * kh * kw, c_out), vec![c_out, c_in, kh, kw]),
        bias: Tensor::new(vec![0.0; c_out], vec![c_out]),
        c_in, c_out, kh, kw, stride: 1, padding: pad, in_h, in_w,
        weight_grad: Rc::new(RefCell::new(vec![0.0; w_len])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; c_out])),
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = v[0];
    for i in 1..v.len() {
        if v[i] > best_val { best_val = v[i]; best = i; }
    }
    best
}

fn evaluate(model: &mut Sequential, images: &[Vec<f32>], raw_labels: &[usize], limit: usize) -> f32 {
    model.set_training(false);
    let mut correct = 0;
    let n = limit.min(images.len());
    for i in 0..n {
        let input = Node::new(images[i].clone(), vec![1, 1, 28, 28]);
        let out = model.forward(input);
        let pred = argmax(&out.borrow().data);
        if pred == raw_labels[i] { correct += 1; }
    }
    model.set_training(true);
    correct as f32 / n as f32
}

fn main() {
    println!("Loading MNIST...");
    let (train_images, n_train, _, _) = read_images("data/mnist/train-images-idx3-ubyte");
    let train_labels = read_labels("data/mnist/train-labels-idx1-ubyte");
    let (test_images, _, _, _) = read_images("data/mnist/t10k-images-idx3-ubyte");
    let test_raw = read_labels_raw("data/mnist/t10k-labels-idx1-ubyte");
    println!("Loaded {} training images", n_train);

    let batch_size = 32;
    let mut loader = DataLoader::new(train_images, train_labels, batch_size);

    let mut model = Sequential {
        list: vec![
            Box::new(conv(1, 8, 3, 3, 1, 28, 28)),
            Box::new(ReLU {}),
            Box::new(MaxPool2d { kernel: 2, stride: 2, channels: 8, in_h: 28, in_w: 28 }),
            Box::new(conv(8, 16, 3, 3, 1, 14, 14)),
            Box::new(ReLU {}),
            Box::new(MaxPool2d { kernel: 2, stride: 2, channels: 16, in_h: 14, in_w: 14 }),
            Box::new(Flatten {}),
            Box::new(Linear {
                weights: Tensor::new(he::he(16 * 7 * 7, 128), vec![16 * 7 * 7, 128]),
                bias: Tensor::new(vec![0.0; 128], vec![128]),
                weights_node: None,
                bias_node: None,
            }),
            Box::new(ReLU {}),
            Box::new(Dropout { probability: 0.2, mask: Vec::new(), training: true }),
            Box::new(Linear {
                weights: Tensor::new(he::he(128, 10), vec![128, 10]),
                bias: Tensor::new(vec![0.0; 10], vec![10]),
                weights_node: None,
                bias_node: None,
            }),
        ],
    };

    let loss_fn = CrossEntropyLoss;
    let mut optimizer = ADAMW {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0001,
        t: 0,
        m: Vec::new(),
        v: Vec::new(),
    };
    let epochs = 15;

    println!("Starting training (AdamW lr={}, batch={}, epochs={})", optimizer.lr, batch_size, epochs);

    for epoch in 0..epochs {
        loader.shuffle();
        let nb = loader.num_batches();
        let mut epoch_loss = 0.0;
        let epoch_start = Instant::now();

        for b in 0..nb {
            let (in_data, tgt_data, bs) = loader.get_batch(b);

            let input = Node::new(in_data, vec![bs, 1, 28, 28]);
            let target = Node::new(tgt_data, vec![bs, 10]);

            model.zero_grad();
            let output = model.forward(input);
            let loss = loss_fn.forward(&output, &target);
            epoch_loss += loss;

            loss_fn.backward(&output, &target);
            model.sync_grads();

            optimizer.step(&mut model.list);

            if b % 100 == 0 {
                println!("  epoch {} batch {}/{} loss {:.4} ({:.1}s elapsed)",
                    epoch, b, nb, loss, epoch_start.elapsed().as_secs_f32());
            }
        }

        let avg = epoch_loss / nb as f32;
        let acc = evaluate(&mut model, &test_images, &test_raw, 1000);
        println!("Epoch {} done: avg loss {:.4}, test accuracy {:.1}% ({:.1}s)",
            epoch, avg, acc * 100.0, epoch_start.elapsed().as_secs_f32());
    }

    println!("Final evaluation on full test set...");
    let final_acc = evaluate(&mut model, &test_images, &test_raw, 10000);
    println!("Final test accuracy: {:.2}%", final_acc * 100.0);
}