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
    use neuralrs::serialize;

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
        // Fan-ins are written as c_in * kh * kw even when c_in is 1.
        #[allow(clippy::identity_op)]
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

        /// Every parameter in the same fixed order, names dropped.
        fn all(&self) -> [&N; 14] {
            self.named().map(|(_, n)| n)
        }

        /// Move every parameter (including BN running stats) to the device once.
        fn to_device(&self) {
            for p in self.all() {
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

        /// Every parameter with its checkpoint name, in a fixed order. Names
        /// follow the PyTorch convention so the file opens cleanly elsewhere.
        fn named(&self) -> [(&'static str, &N); 14] {
            [
                ("conv1.weight", &self.c1w), ("conv1.bias", &self.c1b),
                ("conv2.weight", &self.c2w), ("conv2.bias", &self.c2b),
                ("conv3.weight", &self.c3w), ("conv3.bias", &self.c3b),
                ("fc1.weight", &self.l1w), ("fc1.bias", &self.l1b),
                ("bn.gamma", &self.bg), ("bn.beta", &self.bb),
                ("bn.running_mean", &self.brm), ("bn.running_var", &self.brv),
                ("fc2.weight", &self.l2w), ("fc2.bias", &self.l2b),
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

    /// Saves every parameter (including BN running stats) plus the optimizer
    /// state to a safetensors checkpoint, downloading everything from the
    /// device first. Parameters carry PyTorch-style names; the AdamW state
    /// rides along as `optim.t` and `optim.m.NN`/`optim.v.NN`. Atomic, see
    /// `serialize::save`.
    fn save_checkpoint(p: &Params, optimizer: &AdamW, path: &str) {
        let mut host: Vec<(String, Vec<usize>, Vec<f32>)> = p
            .named()
            .iter()
            .map(|(name, n)| (name.to_string(), n.borrow().shape.clone(), gpu::to_host(n)))
            .collect();

        let (t, moments) = optimizer.export_state();
        host.push(("optim.t".to_string(), vec![1], vec![t as f32]));   // exact for t < 2^24
        let half = moments.len() / 2;
        for (i, m) in moments.into_iter().enumerate() {
            let name = if i < half {
                format!("optim.m.{:02}", i)
            } else {
                format!("optim.v.{:02}", i - half)
            };
            let len = m.len();
            host.push((name, vec![len], m));
        }

        let refs: Vec<serialize::NamedTensor> = host
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        serialize::save(&refs, path);
    }

    /// Loads a checkpoint into freshly built params and, when present, the
    /// optimizer state, everything looked up by name, so tensor order in
    /// the file doesn't matter. Call before `to_device`.
    fn load_checkpoint(p: &Params, optimizer: &mut AdamW, path: &str) {
        let map: std::collections::HashMap<String, Vec<f32>> = serialize::load(path)
            .into_iter()
            .map(|(n, _, d)| (n, d))
            .collect();

        for (name, node) in p.named() {
            let data = map
                .get(name)
                .unwrap_or_else(|| panic!("checkpoint is missing tensor `{name}`"));
            let mut n = node.borrow_mut();
            assert_eq!(data.len(), n.data.len(), "checkpoint tensor `{name}` has the wrong size");
            n.data.copy_from_slice(data);
        }

        if let Some(t_tensor) = map.get("optim.t") {
            let t = t_tensor[0] as usize;
            let mut m_names: Vec<&String> = map.keys().filter(|k| k.starts_with("optim.m.")).collect();
            let mut v_names: Vec<&String> = map.keys().filter(|k| k.starts_with("optim.v.")).collect();
            m_names.sort();
            v_names.sort();
            let moments: Vec<Vec<f32>> = m_names.into_iter().chain(v_names).map(|k| map[k].clone()).collect();
            optimizer.import_state(t, &moments);
            println!("Restored optimizer state (t = {t})");
        }
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

        let ckpt = "mnist_cnn_checkpoint.safetensors";
        let p = Params::init();
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 1e-4);
        if std::path::Path::new(ckpt).exists() {
            load_checkpoint(&p, &mut optimizer, ckpt);
            println!("Resumed from {ckpt}");
        }
        p.to_device();
        let trainable = p.trainable();
        let epochs = 25;

        println!("Training CNN (3 conv + FC/BN/dropout, AdamW lr=0.001, batch={batch_size}, epochs={epochs})");

        for epoch in 0..epochs {
            loader.shuffle();
            let nb = loader.num_batches();
            let mut epoch_loss = 0.0;
            let start = Instant::now();

            for b in 0..nb {
                let (in_data, tgt_data, bs) = loader.get_batch(b);

                let input = Node::new(in_data, vec![bs, 1, 28, 28]);
                let target = Node::new(tgt_data, vec![bs, 10]);
                gpu::to_cuda(&input);
                gpu::to_cuda(&target);
                input.borrow_mut().requires_grad = false;   // image batch is a pure input: skip its gradient

                gpu::zero_grad(&trainable);
                let logits = forward(&input, &p, true);
                let loss = cross_entropy(&logits, &target);
                epoch_loss += loss;

                cross_entropy_backward(&logits, &target);
                backward_graph(&logits);
                optimizer.step(&trainable);

                if b % 100 == 0 {
                    println!("  epoch {epoch} batch {b}/{nb} loss {loss:.4} ({:.1}s)", start.elapsed().as_secs_f32());
                }
            }

            let avg = epoch_loss / nb as f32;
            let acc = evaluate(&test_images, &test_raw, &p, 500);
            println!("Epoch {epoch} done: avg loss {avg:.4}, test acc {:.2}% ({:.1}s)", acc * 100.0, start.elapsed().as_secs_f32());
            save_checkpoint(&p, &optimizer, ckpt);
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