# Changelog

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