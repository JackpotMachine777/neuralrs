# Using NeuralRs

A practical guide to building and training models with this library. It assumes
you know what a neural network is, but not how *this* library expects you to
drive it.

The library gives you two ways to work:

- **CPU**, with `Sequential` and layer structs, close to how PyTorch's `nn.Module` feels
- **GPU** (`--features cuda`), a functional style where you hold parameters yourself and call layer functions

They share one autograd engine, so the concepts transfer. Start on CPU.

---

## 1. The mental model

Three types matter.

**`Tensor`** is storage: data plus a shape. Layers own their weights as tensors.
It does not track gradients by itself.

**`Node`** is a point in the computation graph. It has `data`, `grad`, `shape`,
its `parents`, and a `backward_fn` that knows how to push gradient back to those
parents. Every op you call on a node produces a new node that remembers where it
came from. Nodes are always passed around as `Rc<RefCell<Node>>`.

**The graph** is built implicitly as you run the forward pass. `backward` then
walks it in reverse, filling in every `grad`.

The single most important thing to internalize:

> `backward` fills in `.grad`. It never changes `.data`.
> `optimizer.step` reads `.grad` and changes `.data`.

Everything else follows from that. A training step is always the same five moves:

```
1. zero_grad     clear last step's gradients
2. forward       build the graph, get predictions
3. loss          compare predictions to targets (a number)
4. backward      fill in every .grad
5. step          update the weights using those .grads
```

---

## 2. Your first model (CPU)

Add the dependency:

```toml
[dependencies]
neuralrs = "0.3"
```

A classifier is a `Sequential`: a list of boxed layers, run in order.

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

// helper, because writing the struct literal twice gets old
fn linear(fan_in: usize, fan_out: usize) -> Linear {
    Linear {
        weights: Tensor::new(he::he(fan_in, fan_out), vec![fan_in, fan_out]),
        bias: Tensor::new(vec![0.0; fan_out], vec![fan_out]),
        weights_node: None,
        bias_node: None,
    }
}

let mut model = Sequential {
    list: vec![
        Box::new(linear(20, 64)),
        Box::new(ReLU {}),
        Box::new(linear(64, 3)),   // 3 classes
    ],
};
```

`he::he(fan_in, fan_out)` returns a flat `Vec<f32>` of length `fan_in * fan_out`,
initialized for ReLU networks. Biases start at zero.

Now the loss and the optimizer:

```rust
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
```

Optimizers are plain structs you fill in. `t`, `m`, `v` are internal state and
always start empty.

**`CrossEntropyLoss` takes raw logits.** Do not put a softmax at the end of your
model, the loss does a numerically stable log-softmax internally. If you add one,
you will train a softmax of a softmax and wonder why it learns slowly.

### The training loop

```rust
for epoch in 0..epochs {
    for (inputs, targets, batch) in batches {
        let input  = Node::new(inputs,  vec![batch, 20]);
        let target = Node::new(targets, vec![batch, 3]);   // one-hot

        model.zero_grad();                           // 1
        let output = model.forward(input);           // 2
        let loss = loss_fn.forward(&output, &target);// 3
        loss_fn.backward(&output, &target);          // 4
        model.sync_grads();                          //   <- easy to forget
        optimizer.step(&mut model.list);             // 5
    }
}
```

`model.sync_grads()` copies gradients from the graph nodes back into the layers'
tensors, which is what the optimizer reads. **Skip it and your model silently
never learns**, because every `step` sees zero gradients. It is the most common
mistake with this library.

Targets are one-hot: for class `2` out of 3, the row is `[0.0, 0.0, 1.0]`.

---

## 3. Feeding it your own data

`DataLoader` takes flat `f32` vectors: one per sample.

```rust
use neuralrs::data::dataloader::DataLoader;

// inputs[i] is one sample, flattened. targets[i] is its one-hot label.
let inputs:  Vec<Vec<f32>> = my_samples();
let targets: Vec<Vec<f32>> = my_labels_one_hot();

let mut loader = DataLoader::new(inputs, targets, 32);
```

Then per epoch:

```rust
loader.shuffle();
for b in 0..loader.num_batches() {
    let (in_data, tgt_data, bs) = loader.get_batch(b);
    let input  = Node::new(in_data,  vec![bs, n_features]);
    let target = Node::new(tgt_data, vec![bs, n_classes]);
    // ... training step
}
```

`get_batch` returns **flattened** batches: `in_data` is `bs * n_features` long.
You give the shape when you build the `Node`. The last batch may be smaller than
`batch_size`, which is why `get_batch` hands you `bs`.

**Images** are flattened in `[channels, height, width]` order, so a
28x28 grayscale image is 784 floats and its node shape is `vec![bs, 1, 28, 28]`.
A 32x32 RGB image is 3072 floats: 1024 reds, then 1024 greens, then 1024 blues,
shape `vec![bs, 3, 32, 32]`.

Normalize your inputs. The built-in readers divide pixel bytes by 255.

### Augmentation

```rust
loader.set_augment(Box::new(|img: &[f32]| -> Vec<f32> {
    // called on each training sample, return a modified copy
    img.to_vec()
}));
```

Augmentation runs only in training mode (`loader.set_training(false)` turns it
off for evaluation).

---

## 4. Convolutional models

Conv layers need to know their input size at construction. This is the sharpest
edge in the library: shapes are baked in, not inferred.

```rust
use neuralrs::nn::conv::Conv2d;
use neuralrs::nn::maxpool::MaxPool2d;
use neuralrs::nn::flatten::Flatten;
use std::rc::Rc;
use std::cell::RefCell;

fn conv(c_in: usize, c_out: usize, k: usize, pad: usize, in_h: usize, in_w: usize) -> Conv2d {
    Conv2d {
        weight: Tensor::new(he::he(c_in * k * k, c_out), vec![c_out, c_in, k, k]),
        bias: Tensor::new(vec![0.0; c_out], vec![c_out]),
        c_in, c_out, kh: k, kw: k, stride: 1, padding: pad, in_h, in_w,
        weight_grad: Rc::new(RefCell::new(vec![0.0; c_out * c_in * k * k])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; c_out])),
    }
}
```

You must track the spatial size through the stack yourself:

```rust
let mut model = Sequential {
    list: vec![
        Box::new(conv(1, 16, 3, 1, 28, 28)),   // [1,28,28] -> [16,28,28]
        Box::new(ReLU {}),
        Box::new(MaxPool2d { kernel: 2, stride: 2, channels: 16, in_h: 28, in_w: 28 }),
        //                                                   -> [16,14,14]
        Box::new(conv(16, 32, 3, 1, 14, 14)),  //            -> [32,14,14]
        Box::new(ReLU {}),
        Box::new(MaxPool2d { kernel: 2, stride: 2, channels: 32, in_h: 14, in_w: 14 }),
        //                                                   -> [32,7,7]
        Box::new(Flatten {}),                  //            -> [1568]
        Box::new(linear(32 * 7 * 7, 10)),
    ],
};
```

With `padding: 1` and a 3x3 kernel, the spatial size is unchanged. Each
`MaxPool2d` with kernel 2, stride 2 halves it. Get these numbers wrong and you
get a panic or a wrong-sized matmul, not a silent bug, which is something.

`examples/mnist.rs` is the full version of this with BatchNorm and dropout.

---

## 5. Training vs evaluation mode

Dropout and BatchNorm behave differently at eval time: dropout stops dropping,
BatchNorm uses its running statistics instead of the batch's.

```rust
model.set_training(false);
// ... evaluate ...
model.set_training(true);
```

**Forget this and your test accuracy will be wrong**, usually worse and noisy.
Evaluating with dropout active means you are measuring a randomly crippled model.

Accuracy is just an argmax comparison:

```rust
model.set_training(false);
let output = model.forward(Node::new(test_inputs, vec![n, features]));
let data = output.borrow().data.clone();
let mut correct = 0;
for i in 0..n {
    let row = &data[i * n_classes..(i + 1) * n_classes];
    let pred = row.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    if pred == labels[i] { correct += 1; }
}
println!("accuracy {:.2}%", correct as f32 / n as f32 * 100.0);
model.set_training(true);
```

---

## 6. Saving and loading

```rust
use neuralrs::serialize;

// after training
serialize::save_model(&mut model, "model.txt");

// later, into a model with the *same architecture*
let mut model = build_model();
serialize::load_model(&mut model, "model.txt");
```

For named tensors and PyTorch interoperability, use `save`/`load`, which pick the
format from the file extension:

```rust
let w = &model_weights;   // &[f32]
serialize::save(&[("fc1.weight", &[784, 128], w)], "model.safetensors");

for (name, shape, data) in serialize::load("model.safetensors") {
    println!("{name} {shape:?} ({} values)", data.len());
}
```

`.safetensors` keeps names, shapes, and dtype, and opens in Python:

```python
from safetensors.numpy import load_file
weights = load_file("model.safetensors")
```

Any other extension writes the plain text format instead: no names, no shapes,
but zero dependencies and readable in a diff. Both write to a temp file and
rename, so pressing Ctrl-C mid-save never corrupts the previous checkpoint.

---

## 7. Training on the GPU

The GPU path is functional. There is no `Sequential`: you hold the parameters,
you write the forward pass.

```bash
cargo run --release --features cuda --example mnist_cuda_cnn
```

```rust
use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::cuda::graph::{add, matmul, relu};
use neuralrs::cuda::loss::{cross_entropy, cross_entropy_backward};
use neuralrs::cuda::nn::{conv2d, flatten, maxpool2d};
use neuralrs::cuda::optim::AdamW;
use neuralrs::cuda::runtime as gpu;
```

### Parameters live on the device

Create them as nodes, upload once, reuse every batch:

```rust
let w1 = Node::new(he::he(784, 128), vec![784, 128]);
let b1 = Node::new(vec![0.0; 128], vec![128]);

gpu::to_cuda(&w1);
gpu::to_cuda(&b1);

let trainable = vec![w1.clone(), b1.clone()];
```

After `to_cuda`, the node's CPU-side `data` is empty: the values live in VRAM.
Use `gpu::to_host(&node)` to read them back.

### The forward pass is a function

```rust
fn forward(x: &N, w1: &N, b1: &N, w2: &N, b2: &N) -> N {
    let h = relu(&add(&matmul(x, w1), b1));
    add(&matmul(&h, w2), b2)
}
```

Layers with state take it as arguments:

```rust
// conv2d(input, weight, bias, stride, padding)
let x = conv2d(&x, &conv_w, &conv_b, 1, 1);

// maxpool2d(input, kernel, stride)
let x = maxpool2d(&x, 2, 2);

// batchnorm(input, gamma, beta, running_mean, running_var, momentum, eps, training)
// running stats are nodes too, and are updated in place during training
let x = batchnorm(&x, &gamma, &beta, &run_mean, &run_var, 0.9, 1e-5, training);

// dropout(input, p, training)
let x = dropout(&x, 0.2, training);
```

### The loop

```rust
let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 1e-4);

for b in 0..loader.num_batches() {
    let (in_data, tgt_data, bs) = loader.get_batch(b);

    let input  = Node::new(in_data,  vec![bs, 1, 28, 28]);
    let target = Node::new(tgt_data, vec![bs, 10]);
    gpu::to_cuda(&input);
    gpu::to_cuda(&target);
    input.borrow_mut().requires_grad = false;   // it's a pure input, skip its gradient

    gpu::zero_grad(&trainable);
    let logits = forward(&input, &p, true);
    let loss = cross_entropy(&logits, &target);   // returns f32

    cross_entropy_backward(&logits, &target);     // seeds the output gradient
    backward_graph(&logits);                      // walks the graph
    optimizer.step(&trainable);
}
```

Differences from the CPU loop worth noting:

- Losses come in two halves: `cross_entropy` for the value, `cross_entropy_backward` to seed the gradient. Then you call `backward_graph` yourself.
- There is no `sync_grads`: gradients already live next to the parameters on the device.
- `requires_grad = false` on the input batch skips computing a gradient nobody reads. Free speedup on the first conv layer.
- The optimizer takes `&[Rc<RefCell<Node>>]`, the parameters themselves.

**BatchNorm running statistics are parameters you own, but not ones the optimizer
touches.** They are updated in place by the forward pass. Keep them out of the
list you hand to `optimizer.step`, or the optimizer will treat them as weights
and wreck them.

### Checkpointing with optimizer state

```rust
// save
let (t, moments) = optimizer.export_state();
// ... write params + [t] + moments to safetensors ...

// restore, before the first step()
optimizer.import_state(t, &moments);
```

Without this a resumed run restarts Adam's momentum from zero, which shows up as
a small bump in the loss. `examples/mnist_cuda_cnn.rs` and `examples/cifar_cuda.rs`
both do the full round trip: parameters, running stats, optimizer state, saved
each epoch and restored on startup.

### cuBLAS

`--features cublas` swaps the hand-written GEMM for cuBLAS. It's about 1.9x
faster on large matrices, but only helps if your model is actually matmul-bound.
On a convolutional net it barely moves the epoch time. Measure before you assume.

---

## 8. Choosing CPU or GPU

| | CPU | GPU (`--features cuda`) |
|---|---|---|
| API | `Sequential`, layer structs | functional, you hold the parameters |
| Best for | small models, debugging, no NVIDIA card | anything convolutional, anything big |
| MNIST CNN epoch | minutes | ~3 seconds |
| Gradient checking | easy, everything is on the host | read back with `gpu::read_grad` |

Prototype on CPU with a tiny model, then port. The math is identical, verified by
the test suite: every GPU op is checked against its CPU twin.

---

## 9. When it doesn't work

**Loss doesn't move (CPU).** You forgot `model.sync_grads()`. The optimizer is
stepping on zeros.

**Loss doesn't move (GPU).** Check that `zero_grad` gets the same parameter list
that `optimizer.step` gets, and that every parameter went through `to_cuda`.

**Test accuracy much worse than training.** Either genuine overfitting, or you
forgot `set_training(false)` and are evaluating with dropout on.

**Loss goes to NaN.** Learning rate too high, or you added a softmax before
`CrossEntropyLoss` (it applies one internally). Try `lr = 1e-4`.

**A panic about shapes in conv.** The `in_h`/`in_w` you passed don't match what
the previous layer actually produces. Walk the spatial sizes by hand.

**Gradients look wrong.** Check them numerically. Nudge one weight by `eps`,
measure how the loss changes, compare to the analytic gradient:

```rust
let eps = 1e-3;
// f(w + eps) and f(w - eps), then:
let numeric = (loss_plus - loss_minus) / (2.0 * eps);
// compare against param.grad[i]
```

Agreement to 2-3 decimal places is normal for `f32`. If they agree to 2 digits
but your assertion demands 5, the assertion is wrong, not the code. The test
suite is full of examples of this pattern.

**Branching models on CPU.** `loss_fn.backward` uses a recursive walk that is
correct for straight-line stacks like `Sequential`. If you hand-build a graph
with branches (residual connections, attention), seed the output gradient
yourself and call `backward_graph(&output)` instead, which sorts the graph
topologically first. The GPU path always uses `backward_graph`.

---

## 10. A complete recipe

Training a classifier on your own data, start to finish:

1. **Get your data into `Vec<Vec<f32>>`**: one flat vector per sample, values
   roughly in `[0, 1]` or standardized. Labels as one-hot `Vec<Vec<f32>>`.
2. **Split** into train and test sets before anything else.
3. **Build a small model first.** One hidden layer. Confirm the loss drops on a
   handful of batches. If it doesn't, nothing bigger will help.
4. **Overfit deliberately** on ~100 samples with no dropout. A working setup can
   drive training loss to near zero. If it can't, you have a bug, not a modeling
   problem.
5. **Then scale up**: more layers, dropout, augmentation, more epochs.
6. **Evaluate with `set_training(false)`**, every time.
7. **Checkpoint each epoch** to `.safetensors` so a crash costs one epoch, not the
   whole run.
8. **Move to GPU** when epochs get annoying. The port is mechanical: same math,
   different call style.

The two GPU examples (`mnist_cuda_cnn.rs`, `cifar_cuda.rs`) are the reference for
steps 6-8, checkpointing and resume included. `examples/mnist.rs` is the CPU
reference. Read one of them next: they're short, and everything above is in there.
