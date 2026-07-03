// Flagship CNN on MNIST, fully resident on the GPU.
//
// Same architecture and hyperparameters as the CPU `mnist` example (which hits
// ~99.4%): 3 conv blocks -> FC -> BatchNorm -> ReLU -> Dropout -> FC, AdamW,
// fused cross-entropy, ±1px shift augmentation. Parameters are created once,
// moved to the device once, and reused as the leaves of a fresh graph each
// batch; nothing round-trips to host during training. Run with:
//   cargo run --release --features cuda --example mnist_cuda_cnn

#[cfg(feature = "cuda")]
mod gpu_cnn {
    use neuralrs::autograd::node::{backward_graph, Node};
    use neuralrs::cuda::graph::{add, matmul, relu};
    use neuralrs::cuda::loss::{cross_entropy, cross_entropy_backward};
    use neuralrs::cuda::nn::{batchnorm, conv2d, dropout, flatten, maxpool2d};
    use neuralrs::cuda::optim::AdamW;
    use neuralrs::cuda::runtime as gpu;
    use neuralrs::data::dataloader::DataLoader;
    use neuralrs::data::mnist::{read_images, read_labels, read_labels_raw};
    use neuralrs::init::he;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Instant;

    type N = Rc<RefCell<Node>>;

    /// All model parameters, held as resident leaves reused across batches.
    struct Params {
        c1w: N, c1b: N, c2w: N, c2b: N, c3w: N, c3b: N,
        l1w: N, l1b: N,
        bg: N, bb: N, brm: N, brv: N,
        l2w: N, l2b: N,
    }

    impl Params {
        fn init() -> Self {
            Params {
                c1w: Node::new(he::he(1 * 3 * 3, 16), vec![16, 1, 3, 3]),
                c1b: Node::new(vec![0.0; 16], vec![16]),
                c2w: Node::new(he::he(16 * 3 * 3, 32), vec![32, 16, 3, 3]),
                c2b: Node::new(vec![0.0; 32], vec![32]),
                c3w: Node::new(he::he(32 * 3 * 3, 64), vec![64, 32, 3, 3]),
                c3b: Node::new(vec![0.0; 64], vec![64]),
                l1w: Node::new(he::he(64 * 7 * 7, 128), vec![64 * 7 * 7, 128]),
                l1b: Node::new(vec![0.0; 128], vec![128]),
                bg: Node::new(vec![1.0; 128], vec![128]),
                bb: Node::new(vec![0.0; 128], vec![128]),
                brm: Node::new(vec![0.0; 128], vec![128]),
                brv: Node::new(vec![1.0; 128], vec![128]),
                l2w: Node::new(he::he(128, 10), vec![128, 10]),
                l2b: Node::new(vec![0.0; 10], vec![10]),
            }
        }

        /// Move every parameter (including BN running stats) to the device once.
        fn to_device(&self) {
            for p in [
                &self.c1w, &self.c1b, &self.c2w, &self.c2b, &self.c3w, &self.c3b,
                &self.l1w, &self.l1b, &self.bg, &self.bb, &self.brm, &self.brv,
                &self.l2w, &self.l2b,
            ] {
                gpu::to_cuda(p);
            }
        }

        /// The optimizer-updated parameters (BN running stats are excluded,
        /// they're updated in-place by the BatchNorm forward, not by grads).
        fn trainable(&self) -> Vec<N> {
            vec![
                self.c1w.clone(), self.c1b.clone(), self.c2w.clone(), self.c2b.clone(),
                self.c3w.clone(), self.c3b.clone(), self.l1w.clone(), self.l1b.clone(),
                self.bg.clone(), self.bb.clone(), self.l2w.clone(), self.l2b.clone(),
            ]
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

    /// ±1px random shift augmentation (matches the CPU flagship).
    fn shift_image(img: &[f32]) -> Vec<f32> {
        let size = 28;
        let dx = (rand::random::<f32>() * 3.0).floor() as i32 - 1;
        let dy = (rand::random::<f32>() * 3.0).floor() as i32 - 1;
        let mut out = vec![0.0; size * size];
        for y in 0..size as i32 {
            for x in 0..size as i32 {
                let (sx, sy) = (x - dx, y - dy);
                if sx >= 0 && sx < size as i32 && sy >= 0 && sy < size as i32 {
                    out[(y * size as i32 + x) as usize] = img[(sy * size as i32 + sx) as usize];
                }
            }
        }
        out
    }

    /// The full CNN forward, resident on the GPU.
    fn forward(x: &N, p: &Params, training: bool) -> N {
        // conv block 1: [N,1,28,28] -> [N,16,28,28] -> pool -> [N,16,14,14]
        let x = maxpool2d(&relu(&conv2d(x, &p.c1w, &p.c1b, 1, 1)), 2, 2);
        // conv block 2: -> [N,32,14,14] -> pool -> [N,32,7,7]
        let x = maxpool2d(&relu(&conv2d(&x, &p.c2w, &p.c2b, 1, 1)), 2, 2);
        // conv block 3: -> [N,64,7,7]
        let x = relu(&conv2d(&x, &p.c3w, &p.c3b, 1, 1));
        // flatten -> [N, 3136]
        let x = flatten(&x);
        // FC -> BN -> ReLU -> Dropout -> FC
        let x = add(&matmul(&x, &p.l1w), &p.l1b);
        let x = batchnorm(&x, &p.bg, &p.bb, &p.brm, &p.brv, 0.9, 1e-5, training);
        let x = relu(&x);
        let x = dropout(&x, 0.2, training);
        add(&matmul(&x, &p.l2w), &p.l2b)
    }

    fn evaluate(test_images: &[Vec<f32>], test_raw: &[usize], p: &Params, chunk: usize) -> f32 {
        let n = test_images.len();
        let mut correct = 0;
        let mut i = 0;
        while i < n {
            let end = (i + chunk).min(n);
            let bs = end - i;
            let mut flat = Vec::with_capacity(bs * 28 * 28);
            for img in &test_images[i..end] { flat.extend_from_slice(img); }
            let input = Node::new(flat, vec![bs, 1, 28, 28]);
            gpu::to_cuda(&input);
            let logits = forward(&input, p, false);
            let data = gpu::to_host(&logits);
            for j in 0..bs {
                if argmax(&data[j * 10..j * 10 + 10]) == test_raw[i + j] { correct += 1; }
            }
            i = end;
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

        let batch_size = 32;
        let mut loader = DataLoader::new(train_images, train_labels, batch_size);
        loader.set_augment(Box::new(shift_image));

        let p = Params::init();
        p.to_device();
        let trainable = p.trainable();

        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 1e-4);
        let epochs = 25;

        println!("Training CNN (3 conv + FC/BN/dropout, AdamW lr=0.001, batch={batch_size}, epochs={epochs})");

        for epoch in 0..epochs {
            loader.shuffle();
            let nb = loader.num_batches();
            let mut epoch_loss = 0.0;
            let (mut t_fwd, mut t_bwd, mut t_step) = (0.0f32, 0.0f32, 0.0f32);
            let start = Instant::now();

            for b in 0..nb {
                let (in_data, tgt_data, bs) = loader.get_batch(b);

                let input = Node::new(in_data, vec![bs, 1, 28, 28]);
                let target = Node::new(tgt_data, vec![bs, 10]);
                gpu::to_cuda(&input);
                gpu::to_cuda(&target);
                input.borrow_mut().requires_grad = false;

                gpu::zero_grad(&trainable);

                gpu::synchronize();
                let t0 = Instant::now();
                let logits = forward(&input, &p, true);
                let loss = cross_entropy(&logits, &target);
                gpu::synchronize();
                let t1 = Instant::now();

                cross_entropy_backward(&logits, &target);
                backward_graph(&logits);
                gpu::synchronize();
                let t2 = Instant::now();

                optimizer.step(&trainable);
                gpu::synchronize();
                let t3 = Instant::now();

                t_fwd += (t1 - t0).as_secs_f32();
                t_bwd += (t2 - t1).as_secs_f32();
                t_step += (t3 - t2).as_secs_f32();
                epoch_loss += loss;

                if b % 100 == 0 {
                    println!("  epoch {epoch} batch {b}/{nb} loss {loss:.4} ({:.1}s)", start.elapsed().as_secs_f32());
                }
            }

            let avg = epoch_loss / nb as f32;
            let acc = evaluate(&test_images, &test_raw, &p, 500);
            let secs = start.elapsed().as_secs_f32();
            let imgs = 60000.0 / secs;
            println!("Epoch {epoch} done: avg loss {avg:.4}, test acc {:.2}% ({secs:.1}s, {imgs:.0} img/s)", acc * 100.0);
            println!("  phases: fwd {t_fwd:.2}s | bwd {t_bwd:.2}s | step {t_step:.2}s");
        }

        println!("Final evaluation on full test set...");
        let acc = evaluate(&test_images, &test_raw, &p, 500);
        println!("Final test accuracy: {:.2}%", acc * 100.0);
    }
}

#[cfg(feature = "cuda")]
fn main() {
    gpu_cnn::run();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the cuda feature:");
    println!("  cargo run --release --features cuda --example mnist_cuda_cnn");
}