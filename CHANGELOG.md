# Changelog

## 0.3.0

- **Full GPU layer parity**: resident `BatchNorm2d` and `Dropout2d` complete the
  CUDA backend, every layer in the library now has a device twin, each
  gradient-checked against its CPU implementation
- **Optimizer-state checkpointing**: `AdamW::export_state`/`import_state` move
  the step counter and both moment buffers to and from the host, so a resumed
  run continues with bias correction and momentum intact
- **safetensors support**: `serialize::save`/`load` pick the on-disk format from
  the file extension, `.safetensors` (binary, names and shapes preserved,
  loadable from PyTorch and the wider ecosystem) or the existing positional text
  format. Both paths write atomically
- **CIFAR-10**: a binary-format loader (`data::cifar`) and a GPU-resident
  VGG-style example (3.3M parameters, ~90% test accuracy in 40 epochs)
- **Optional cuBLAS backend** (`--features cublas`): routes the resident matmul
  through cuBLAS, about 1.9x the hand-written kernel on a 1024^3 GEMM
  (13.9 vs 7.2 TFLOP/s). The own register-blocked kernel stays the default

## 0.2.0

- Complete resident CUDA/GPU backend (`--features cuda`): every graph op,
  activation, loss, and optimizer, plus the full layer stack up to
  `TransformerBlock`, each gradient-checked against the CPU implementation
- Flagship CNN trains fully on the GPU to 99.43% on MNIST (~19k images/s
  on an RTX 5060 Ti)
- conv2d lowered to im2col + register-blocked GEMM (forward, input grad,
  split-K weight grad); resident matmul register-blocked (~4.5x on large
  matrices)
- `Node::requires_grad`: pure-input leaves can skip gradient computation
- Atomic tensor checkpointing (`serialize::save_tensors`/`load_tensors`);
  the GPU flagship saves each epoch and resumes on startup
- Removed the transitional `cuda::matmul` module and the empty `dispatch`
  stub

## 0.1.0

- Initial release: the CPU stack: autograd, layers up to
  `TransformerBlock`, optimizers, schedulers, MNIST CNN at 99.44%