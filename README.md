# NeuralRs

A deep learning library written from scratch in Rust — with its own autograd engine, a full neural network stack, and a working Transformer. Not a toy: it trains real models, every layer's gradient is numerically verified, and it reaches **99.44% on MNIST**.

NeuralRs mirrors the design of PyTorch (tensors, autograd, `nn` modules, optimizers) but is built end-to-end in Rust with minimal dependencies. The goal is a framework that is both **complete enough to be useful** and **readable enough to learn from** — if you've ever wanted to see how a deep learning framework actually works under the hood, the source is meant to be read.

## Why NeuralRs?

- **Built from scratch.** The autograd engine, every layer, every optimizer — all implemented by hand. No `libtorch` bindings, no autodiff crate. The only heavy lifting borrowed is an optional BLAS backend (off by default).
- **Actually complete.** Most "I wrote a DL framework in Rust" projects stop at an MLP. NeuralRs has convolutions, recurrent layers (RNN/LSTM), and a full batched Transformer block — all sharing one autograd engine.
- **Verified, not hoped.** Every layer's backward pass is checked against numerical gradients. The Transformer, attention, batch-norm, conv — all gradient-checked in the test suite.
- **Fast where it counts.** SIMD (AVX2) matmul, Rayon-parallel convolution and elementwise ops, and an optional pure-Rust BLAS backend via a feature flag.
- **Readable.** The code is meant to be followed. If you want to understand how backprop, attention, or a conv layer really works, you can read the implementation top to bottom.

## Features

**Autograd**
- Dynamic computation graph with reverse-mode automatic differentiation
- Topological backward pass (handles branching graphs like attention and residual connections)
- Shape-agnostic tensors (2D, 3D, 4D)

**Layers**
- `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`
- `Dropout`, `Dropout2d`
- `BatchNorm` (1D), `BatchNorm2d` (spatial, for conv feature maps), `LayerNorm`
- `RNN`, `LSTM`
- `MultiHeadAttention`, positional encoding, and a full **`TransformerBlock`** — all with batch support

**Activations**
- ReLU, LeakyReLU, ELU, SiLU, GELU, Sigmoid, Tanh

**Optimizers**
- SGD, SGD with Nesterov momentum, Adam, AdamW, RMSprop, Adagrad, NAdam

**Learning rate schedulers**
- StepLR, ExponentialLR, CosineAnnealing, WarmupCosine

**Performance**
- AVX2 SIMD matmul
- Rayon-parallel convolution (forward and backward) and elementwise operations
- Optional BLAS backend (`--features blas`) — your own SIMD matmul stays the default

**Utilities**
- `DataLoader` with shuffling and data augmentation
- Model save / load
- Cross-entropy (with built-in stable log-softmax) and MSE losses

## Quick start

Build a model the way you would in PyTorch — stack layers in a `Sequential`, then run a standard forward / backward / step loop:

```rust
use neuralrs::nn::sequential::Sequential;
use neuralrs::nn::linear::Linear;
use neuralrs::nn::activations::relu::ReLU;
use neuralrs::nn::module::Module;
use neuralrs::nn::loss::{Loss, CrossEntropyLoss};
use neuralrs::optim::adamw::ADAMW;
use neuralrs::autograd::node::Node;
use neuralrs::tensor::Tensor;
use neuralrs::init::he;

// A small MLP: 784 -> 128 -> 10
let mut model = Sequential {
    list: vec![
        Box::new(Linear {
            weights: Tensor::new(he::he(784, 128), vec![784, 128]),
            bias: Tensor::new(vec![0.0; 128], vec![128]),
            weights_node: None,
            bias_node: None,
        }),
        Box::new(ReLU {}),
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

// Training step (inputs: [batch, 784], targets: one-hot [batch, 10])
let input = Node::new(batch_inputs, vec![batch_size, 784]);
let target = Node::new(batch_targets, vec![batch_size, 10]);

model.zero_grad();
let output = model.forward(input);
let loss = loss_fn.forward(&output, &target); // takes raw logits
loss_fn.backward(&output, &target);
model.sync_grads();
optimizer.step(&mut model.list);
```

See [`examples/mnist.rs`](examples/mnist.rs) for a complete training loop: a 3-layer CNN with BatchNorm, dropout, data augmentation, and AdamW. The [`examples/mnist_sched.rs`](examples/mnist_sched.rs) variant adds WarmupCosine learning-rate scheduling.

## Running the examples

```bash
# Train a CNN on MNIST (reaches ~99.4%)
cargo run --release --example mnist

# Compact 2-conv variant (faster, ~99.3%)
cargo run --release --example mnist_compact

# With the optional BLAS backend
cargo run --release --features blas --example mnist
```

MNIST data (IDX files) goes in `data/mnist/`.

## Benchmarks

On MNIST (28×28 grayscale, 10 classes):

| Model | Setup | Test accuracy |
|-------|-------|---------------|
| 3-conv CNN (16→32→64) | AdamW, augmentation, 25 epochs | **99.44%** |
| 2-conv CNN (8→16) | AdamW, WarmupCosine, 25 epochs | 99.39% |
| 2-conv CNN (8→16) | AdamW, 25 epochs | 99.29% |

## Project status

NeuralRs is an actively developed personal project. The full training stack works today — CNNs, RNNs, and Transformers all train and their gradients are verified. Contributions, issues, and ideas are welcome.

### Roadmap

- [ ] CUDA / GPU backend (the real performance frontier)
- [ ] More example models (a Transformer training demo)
- [ ] Expanded documentation

## Design philosophy

The core is 100% hand-written Rust — the autograd engine and the math that makes a deep learning framework a deep learning framework. Performance shortcuts that don't belong to that core (like a BLAS backend) are optional and opt-in, never the default. The point is to own the important parts, not to wrap someone else's library.

## License

Apache-2.0