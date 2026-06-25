// MLP on MNIST, fully resident on the GPU.
//
// 784 -> 128 -> ReLU -> 10, AdamW, fused cross-entropy. Parameters are created
// once, moved to the device once, and reused as the leaves of a fresh graph each
// batch, nothing round-trips to host during training. Run with:
//   cargo run --release --features cuda --example mnist_cuda

#[cfg(feature = "cuda")]
mod gpu_mnist {
    use neuralrs::autograd::node::{backward_graph, Node};
    use neuralrs::cuda::graph::{add, matmul, relu};
    use neuralrs::cuda::loss::{cross_entropy, cross_entropy_backward};
    use neuralrs::cuda::optim::AdamW;
    use neuralrs::cuda::runtime as gpu;
    use neuralrs::data::dataloader::DataLoader;
    use neuralrs::data::mnist::{read_images, read_labels, read_labels_raw};
    use neuralrs::init::he;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Instant;

    type N = Rc<RefCell<Node>>;

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        let mut best_val = v[0];
        for i in 1..v.len() {
            if v[i] > best_val { best_val = v[i]; best = i; }
        }
        best
    }

    fn forward(x: &N, w1: &N, b1: &N, w2: &N, b2: &N) -> N {
        let h = relu(&add(&matmul(x, w1), b1));
        add(&matmul(&h, w2), b2)
    }

    fn evaluate(test_images: &[Vec<f32>], test_raw: &[usize], w1: &N, b1: &N, w2: &N, b2: &N) -> f32 {
        let n = test_images.len();
        let mut flat = Vec::with_capacity(n * 784);
        for img in test_images { flat.extend_from_slice(img); }

        let input = Node::new(flat, vec![n, 784]);
        gpu::to_cuda(&input);
        let logits = forward(&input, w1, b1, w2, b2);
        let data = gpu::to_host(&logits);

        let mut correct = 0;
        for i in 0..n {
            if argmax(&data[i * 10..i * 10 + 10]) == test_raw[i] { correct += 1; }
        }
        correct as f32 / n as f32
    }

    pub fn run() {
        println!("Loading MNIST...");
        let (train_images, n_train, _, _) = read_images("data/mnist/train-images-idx3-ubyte");
        let train_labels = read_labels("data/mnist/train-labels-idx1-ubyte");
        let (test_images, _, _, _) = read_images("data/mnist/t10k-images-idx3-ubyte");
        let test_raw = read_labels_raw("data/mnist/t10k-labels-idx1-ubyte");
        println!("Loaded {n_train} training images");

        let batch_size = 128;
        let mut loader = DataLoader::new(train_images, train_labels, batch_size);

        let w1 = Node::new(he::he(784, 128), vec![784, 128]);
        let b1 = Node::new(vec![0.0; 128], vec![128]);
        let w2 = Node::new(he::he(128, 10), vec![128, 10]);
        let b2 = Node::new(vec![0.0; 10], vec![10]);
        for p in [&w1, &b1, &w2, &b2] { gpu::to_cuda(p); }
        let params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];

        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 1e-4);
        let epochs = 5;

        println!("Training MLP 784->128->10 (AdamW lr=0.001, batch={batch_size}, epochs={epochs})");

        for epoch in 0..epochs {
            loader.shuffle();
            let nb = loader.num_batches();
            let mut epoch_loss = 0.0;
            let start = Instant::now();

            for b in 0..nb {
                let (in_data, tgt_data, bs) = loader.get_batch(b);

                let input = Node::new(in_data, vec![bs, 784]);
                let target = Node::new(tgt_data, vec![bs, 10]);
                gpu::to_cuda(&input);
                gpu::to_cuda(&target);

                gpu::zero_grad(&params);
                let logits = forward(&input, &w1, &b1, &w2, &b2);
                let loss = cross_entropy(&logits, &target);
                epoch_loss += loss;

                cross_entropy_backward(&logits, &target);
                backward_graph(&logits);
                optimizer.step(&params);

                if b % 100 == 0 {
                    println!("  epoch {epoch} batch {b}/{nb} loss {loss:.4} ({:.1}s)", start.elapsed().as_secs_f32());
                }
            }

            let avg = epoch_loss / nb as f32;
            let acc = evaluate(&test_images, &test_raw, &w1, &b1, &w2, &b2);
            println!("Epoch {epoch} done: avg loss {avg:.4}, test acc {:.2}% ({:.1}s)", acc * 100.0, start.elapsed().as_secs_f32());
        }

        let acc = evaluate(&test_images, &test_raw, &w1, &b1, &w2, &b2);
        println!("Final test accuracy: {:.2}%", acc * 100.0);
    }
}

#[cfg(feature = "cuda")]
fn main() {
    gpu_mnist::run();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the cuda feature:");
    println!("  cargo run --release --features cuda --example mnist_cuda");
}