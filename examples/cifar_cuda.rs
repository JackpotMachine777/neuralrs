// CIFAR-10 CNN, fully resident on the GPU (~5.5M params).
//
// A VGG-style net: three blocks of two 3x3 convolutions (64 -> 128 -> 256
// channels) each followed by BatchNorm2d + ReLU and a 2x2 max-pool, then
// FC -> BatchNorm -> ReLU -> Dropout -> FC. Everything lives on the device;
// the only host round-trips are batch upload and the loss/accuracy readback.
// Trains with AdamW, random-flip + shift augmentation, and checkpoints to
// safetensors each epoch (resuming automatically, optimizer state included).
// Run with:
//   cargo run --release --features cuda --example cifar_cuda

#[cfg(feature = "cuda")]
mod gpu_cifar {
    use neuralrs::autograd::node::{backward_graph, Node};
    use neuralrs::cuda::graph::{add, matmul, relu};
    use neuralrs::cuda::loss::{cross_entropy, cross_entropy_backward};
    use neuralrs::cuda::nn::{batchnorm, batchnorm2d, conv2d, dropout, flatten, maxpool2d};
    use neuralrs::cuda::optim::AdamW;
    use neuralrs::cuda::runtime as gpu;
    use neuralrs::data::cifar::{read_images, read_labels, read_labels_raw};
    use neuralrs::data::dataloader::DataLoader;
    use neuralrs::init::he;
    use neuralrs::serialize;

    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Instant;

    type N = Rc<RefCell<Node>>;

    fn conv_w(c_out: usize, c_in: usize) -> N {
        Node::new(he::he(c_in * 3 * 3, c_out), vec![c_out, c_in, 3, 3])
    }
    fn zeros(n: usize) -> N { Node::new(vec![0.0; n], vec![n]) }
    fn ones(n: usize) -> N { Node::new(vec![1.0; n], vec![n]) }

    /// One conv + BatchNorm2d unit: conv weight/bias and the BN2d gamma/beta
    /// plus running stats.
    struct ConvBN { w: N, b: N, g: N, bt: N, rm: N, rv: N }
    impl ConvBN {
        fn new(c_out: usize, c_in: usize) -> Self {
            ConvBN { w: conv_w(c_out, c_in), b: zeros(c_out), g: ones(c_out), bt: zeros(c_out), rm: zeros(c_out), rv: ones(c_out) }
        }
        fn forward(&self, x: &N, training: bool) -> N {
            let c = conv2d(x, &self.w, &self.b, 1, 1);
            let n = batchnorm2d(&c, &self.g, &self.bt, &self.rm, &self.rv, 0.9, 1e-5, training);
            relu(&n)
        }
        fn named(&self, prefix: &str) -> Vec<(String, &N)> {
            vec![
                (format!("{prefix}.weight"), &self.w), (format!("{prefix}.bias"), &self.b),
                (format!("{prefix}.bn.gamma"), &self.g), (format!("{prefix}.bn.beta"), &self.bt),
                (format!("{prefix}.bn.running_mean"), &self.rm), (format!("{prefix}.bn.running_var"), &self.rv),
            ]
        }
        fn trainable(&self) -> Vec<N> {
            vec![self.w.clone(), self.b.clone(), self.g.clone(), self.bt.clone()]
        }
    }

    struct Params {
        c1a: ConvBN, c1b: ConvBN,   // block 1: 3->64, 64->64
        c2a: ConvBN, c2b: ConvBN,   // block 2: 64->128, 128->128
        c3a: ConvBN, c3b: ConvBN,   // block 3: 128->256, 256->256
        l1w: N, l1b: N,             // FC 4096->512
        bg: N, bb: N, brm: N, brv: N,
        l2w: N, l2b: N,             // FC 512->10
    }

    impl Params {
        fn init() -> Self {
            Params {
                c1a: ConvBN::new(64, 3),    c1b: ConvBN::new(64, 64),
                c2a: ConvBN::new(128, 64),  c2b: ConvBN::new(128, 128),
                c3a: ConvBN::new(256, 128), c3b: ConvBN::new(256, 256),
                l1w: Node::new(he::he(256 * 4 * 4, 512), vec![256 * 4 * 4, 512]),
                l1b: zeros(512),
                bg: ones(512), bb: zeros(512), brm: zeros(512), brv: ones(512),
                l2w: Node::new(he::he(512, 10), vec![512, 10]),
                l2b: zeros(10),
            }
        }

        /// Every parameter with its checkpoint name, in a fixed order.
        fn named(&self) -> Vec<(String, &N)> {
            let mut v = Vec::new();
            v.extend(self.c1a.named("conv1a")); v.extend(self.c1b.named("conv1b"));
            v.extend(self.c2a.named("conv2a")); v.extend(self.c2b.named("conv2b"));
            v.extend(self.c3a.named("conv3a")); v.extend(self.c3b.named("conv3b"));
            v.extend(vec![
                ("fc1.weight".to_string(), &self.l1w), ("fc1.bias".to_string(), &self.l1b),
                ("bn.gamma".to_string(), &self.bg), ("bn.beta".to_string(), &self.bb),
                ("bn.running_mean".to_string(), &self.brm), ("bn.running_var".to_string(), &self.brv),
                ("fc2.weight".to_string(), &self.l2w), ("fc2.bias".to_string(), &self.l2b),
            ]);
            v
        }

        fn to_device(&self) {
            for (_, n) in self.named() { gpu::to_cuda(n); }
        }

        /// Optimizer-updated params (BN running stats excluded, updated in
        /// place by the BN forward, not by gradients).
        fn trainable(&self) -> Vec<N> {
            let mut v = Vec::new();
            for cb in [&self.c1a, &self.c1b, &self.c2a, &self.c2b, &self.c3a, &self.c3b] {
                v.extend(cb.trainable());
            }
            v.extend(vec![
                self.l1w.clone(), self.l1b.clone(),
                self.bg.clone(), self.bb.clone(),
                self.l2w.clone(), self.l2b.clone(),
            ]);
            v
        }
    }

    fn forward(x: &N, p: &Params, training: bool) -> N {
        // block 1 -> [64,16,16]
        let x = p.c1a.forward(x, training);
        let x = maxpool2d(&p.c1b.forward(&x, training), 2, 2);
        // block 2 -> [128,8,8]
        let x = p.c2a.forward(&x, training);
        let x = maxpool2d(&p.c2b.forward(&x, training), 2, 2);
        // block 3 -> [256,4,4]
        let x = p.c3a.forward(&x, training);
        let x = maxpool2d(&p.c3b.forward(&x, training), 2, 2);
        // head
        let x = flatten(&x);
        let x = add(&matmul(&x, &p.l1w), &p.l1b);
        let x = batchnorm(&x, &p.bg, &p.bb, &p.brm, &p.brv, 0.9, 1e-5, training);
        let x = relu(&x);
        let x = dropout(&x, 0.5, training);
        add(&matmul(&x, &p.l2w), &p.l2b)
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        let mut bv = v[0];
        for i in 1..v.len() {
            if v[i] > bv { bv = v[i]; best = i; }
        }
        best
    }

    /// Random horizontal flip + up-to-4px shift with zero padding, on a
    /// [3,32,32] image. Standard CIFAR augmentation.
    fn augment(img: &[f32]) -> Vec<f32> {
        let (c, s) = (3usize, 32usize);
        let flip = rand::random::<bool>();
        let dx = (rand::random::<f32>() * 9.0) as i32 - 4;
        let dy = (rand::random::<f32>() * 9.0) as i32 - 4;
        let mut out = vec![0.0; c * s * s];
        for ch in 0..c {
            for y in 0..s as i32 {
                for x in 0..s as i32 {
                    let sx = x - dx;
                    let sy = y - dy;
                    if sx >= 0 && sx < s as i32 && sy >= 0 && sy < s as i32 {
                        let src_x = if flip { s as i32 - 1 - sx } else { sx };
                        out[(ch * s + y as usize) * s + x as usize] =
                            img[(ch * s + sy as usize) * s + src_x as usize];
                    }
                }
            }
        }
        out
    }

    fn evaluate(test_images: &[Vec<f32>], test_raw: &[usize], p: &Params, chunk: usize) -> f32 {
        let n = test_images.len();
        let mut correct = 0;
        let mut i = 0;
        while i < n {
            let end = (i + chunk).min(n);
            let bs = end - i;
            let mut flat = Vec::with_capacity(bs * 3072);
            for img in &test_images[i..end] { flat.extend_from_slice(img); }
            let input = Node::new(flat, vec![bs, 3, 32, 32]);
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

    fn save_checkpoint(p: &Params, optimizer: &AdamW, path: &str) {
        let mut host: Vec<(String, Vec<usize>, Vec<f32>)> = p
            .named()
            .iter()
            .map(|(name, n)| (name.clone(), n.borrow().shape.clone(), gpu::to_host(n)))
            .collect();
        let (t, moments) = optimizer.export_state();
        host.push(("optim.t".to_string(), vec![1], vec![t as f32]));
        let half = moments.len() / 2;
        for (i, m) in moments.into_iter().enumerate() {
            let name = if i < half { format!("optim.m.{:03}", i) } else { format!("optim.v.{:03}", i - half) };
            let len = m.len();
            host.push((name, vec![len], m));
        }
        let refs: Vec<serialize::NamedTensor> = host.iter().map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice())).collect();
        serialize::save(&refs, path);
    }

    fn load_checkpoint(p: &Params, optimizer: &mut AdamW, path: &str) {
        let map: std::collections::HashMap<String, Vec<f32>> =
            serialize::load(path).into_iter().map(|(n, _, d)| (n, d)).collect();
        for (name, node) in p.named() {
            let data = map.get(&name).unwrap_or_else(|| panic!("checkpoint missing tensor `{name}`"));
            let mut n = node.borrow_mut();
            assert_eq!(data.len(), n.data.len(), "checkpoint tensor `{name}` wrong size");
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
        println!("Loading CIFAR-10...");
        let train_files: Vec<String> = (1..=5).map(|i| format!("data/cifar-10-batches-bin/data_batch_{i}.bin")).collect();
        let train_refs: Vec<&str> = train_files.iter().map(|s| s.as_str()).collect();
        let train_images = read_images(&train_refs);
        let train_labels = read_labels(&train_refs);
        let test_images = read_images(&["data/cifar-10-batches-bin/test_batch.bin"]);
        let test_raw = read_labels_raw(&["data/cifar-10-batches-bin/test_batch.bin"]);
        println!("Loaded {} training, {} test images", train_images.len(), test_images.len());

        let batch_size = 128;
        let mut loader = DataLoader::new(train_images, train_labels, batch_size);
        loader.set_augment(Box::new(augment));

        let ckpt = "cifar_cnn_checkpoint.safetensors";
        let p = Params::init();
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 5e-4);
        if std::path::Path::new(ckpt).exists() {
            load_checkpoint(&p, &mut optimizer, ckpt);
            println!("Resumed from {ckpt}");
        }
        p.to_device();
        let trainable = p.trainable();
        let n_params: usize = p.named().iter()
            .filter(|(n, _)| !n.contains("running_"))
            .map(|(_, node)| node.borrow().data.len().max(node.borrow().shape.iter().product()))
            .sum();
        println!("Model: ~{:.1}M parameters, batch {batch_size}", n_params as f32 / 1e6);

        let epochs = 40;
        for epoch in 0..epochs {
            loader.shuffle();
            let nb = loader.num_batches();
            let mut epoch_loss = 0.0;
            let start = Instant::now();

            for b in 0..nb {
                let (in_data, tgt_data, bs) = loader.get_batch(b);
                let input = Node::new(in_data, vec![bs, 3, 32, 32]);
                let target = Node::new(tgt_data, vec![bs, 10]);
                gpu::to_cuda(&input);
                gpu::to_cuda(&target);
                input.borrow_mut().requires_grad = false;

                gpu::zero_grad(&trainable);
                let logits = forward(&input, &p, true);
                let loss = cross_entropy(&logits, &target);
                epoch_loss += loss;
                cross_entropy_backward(&logits, &target);
                backward_graph(&logits);
                optimizer.step(&trainable);

                if b % 50 == 0 {
                    println!("  epoch {epoch} batch {b}/{nb} loss {loss:.4} ({:.1}s)", start.elapsed().as_secs_f32());
                }
            }

            let avg = epoch_loss / nb as f32;
            let acc = evaluate(&test_images, &test_raw, &p, 250);
            println!("Epoch {epoch} done: avg loss {avg:.4}, test acc {:.2}% ({:.1}s)", acc * 100.0, start.elapsed().as_secs_f32());
            save_checkpoint(&p, &optimizer, ckpt);
        }

        println!("Final test accuracy: {:.2}%", evaluate(&test_images, &test_raw, &p, 250) * 100.0);
    }
}

#[cfg(feature = "cuda")]
fn main() {
    gpu_cifar::run();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the cuda feature:");
    println!("  cargo run --release --features cuda --example cifar_cuda");
}