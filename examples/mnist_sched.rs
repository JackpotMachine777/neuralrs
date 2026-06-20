use neuralrs::tensor::Tensor;
use neuralrs::nn::module::Module;
use neuralrs::nn::sequential::Sequential;
use neuralrs::nn::conv::Conv2d;
use neuralrs::nn::maxpool::MaxPool2d;
use neuralrs::nn::flatten::Flatten;
use neuralrs::nn::linear::Linear;
use neuralrs::nn::activations::relu::ReLU;
use neuralrs::nn::loss::{Loss, CrossEntropyLoss};
use neuralrs::data::dataloader::DataLoader;
use neuralrs::data::mnist::{read_images, read_labels, read_labels_raw};
use neuralrs::autograd::node::Node;
use neuralrs::init::he;
use neuralrs::optim::adamw::ADAMW;
use neuralrs::optim::scheduler::{Scheduler, WarmupCosine};
use neuralrs::nn::dropout::Dropout;
use neuralrs::nn::batchnorm::BatchNorm;

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

fn shift_image(img: &[f32]) -> Vec<f32> {
    let size = 28;
    let dx = (rand::random::<f32>() * 3.0).floor() as i32 - 1;
    let dy = (rand::random::<f32>() * 3.0).floor() as i32 - 1;

    let mut out = vec![0.0; size * size];
    for y in 0..size as i32 {
        for x in 0..size as i32 {
            let src_x = x - dx;
            let src_y = y - dy;
            if src_x >= 0 && src_x < size as i32 && src_y >= 0 && src_y < size as i32 {
                out[(y * size as i32 + x) as usize] = img[(src_y * size as i32 + src_x) as usize];
            }
        }
    }
    out
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
    println!("Loaded {n_train} training images");

    let batch_size = 32;
    let mut loader = DataLoader::new(train_images, train_labels, batch_size);
    loader.set_augment(Box::new(shift_image));

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
            Box::new(BatchNorm {
                gamma: Tensor::new(vec![1.0; 128], vec![128]),
                beta: Tensor::new(vec![0.0; 128], vec![128]),
                epsilon: 1e-5,
                num_features: 128,
                gamma_grad: Rc::new(RefCell::new(vec![0.0; 128])),
                beta_grad: Rc::new(RefCell::new(vec![0.0; 128])),
                running_mean: vec![0.0; 128],
                running_var: vec![1.0; 128],
                momentum: 0.9,
                training: true,
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
        lr: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8,
        weight_decay: 0.0001, t: 0, m: Vec::new(), v: Vec::new(),
    };
    let epochs = 25;

    let scheduler = WarmupCosine {
        base_lr: 0.001,
        min_lr: 0.00001,
        warmup_steps: 2,
        t_max: epochs,
    };

    println!("Starting training (small model + AdamW + WarmupCosine, batch={batch_size}, epochs={epochs})");

    for epoch in 0..epochs {
        let lr = scheduler.get_lr(epoch);
        optimizer.lr = lr;

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
                println!("  epoch {} batch {}/{} loss {:.4} ({:.1}s)",
                    epoch, b, nb, loss, epoch_start.elapsed().as_secs_f32());
            }
        }

        let avg = epoch_loss / nb as f32;
        let acc = evaluate(&mut model, &test_images, &test_raw, 1000);
        println!("Epoch {} done: lr {:.6}, avg loss {:.4}, test acc {:.1}% ({:.1}s)",
            epoch, lr, avg, acc * 100.0, epoch_start.elapsed().as_secs_f32());
    }

    println!("Final evaluation on full test set...");
    let final_acc = evaluate(&mut model, &test_images, &test_raw, 10000);
    println!("Final test accuracy: {:.2}%", final_acc * 100.0);
}