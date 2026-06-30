//! Embedding lookup table on the GPU (resident). Maps token indices to rows of a
//! resident weight matrix [vocab, dim]; backward scatters the gradient back to
//! the looked-up rows with atomicAdd (repeated tokens accumulate), writing
//! directly into the weight's resident gradient. Mirrors the CPU `Embedding`.

use std::cell::RefCell;
use std::rc::Rc;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::autograd::node::{GpuBuffers, Node};
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void embed_forward(float* out, const float* weight, const int* indices, int seq_len, int dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = seq_len * dim;
        if (idx >= total) return;
        int pos = idx / dim;
        int d = idx % dim;
        int tok = indices[pos];
        out[pos * dim + d] = weight[tok * dim + d];
    }
    extern "C" __global__ void embed_backward(float* w_grad, const float* grad, const int* indices, int seq_len, int dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = seq_len * dim;
        if (idx >= total) return;
        int pos = idx / dim;
        int d = idx % dim;
        int tok = indices[pos];
        atomicAdd(&w_grad[tok * dim + d], grad[pos * dim + d]);
    }
"#;
crate::kernel_module!(KERNEL);

/// An embedding table over a resident weight matrix `[vocab_size, embedding_dim]`.
pub struct Embedding {
    pub weight: Rc<RefCell<Node>>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    /// Wraps a resident weight node `[vocab_size, embedding_dim]` as an embedding.
    pub fn new(weight: Rc<RefCell<Node>>, vocab_size: usize, embedding_dim: usize) -> Self {
        Self { weight, vocab_size, embedding_dim }
    }

    /// Looks up `indices` (token IDs), returning a resident `[seq_len, dim]` node.
    /// Backward scatters into the weight's gradient.
    ///
    /// # Panics
    /// If the weight is not on the GPU, or any index is out of vocab range.
    pub fn forward(&self, indices: &[usize]) -> Rc<RefCell<Node>> {
        let stream = backend::stream();
        let seq_len = indices.len();
        let dim = self.embedding_dim;
        let total = seq_len * dim;

        let idx_i32: Vec<i32> = indices.iter().map(|&i| {
            assert!(i < self.vocab_size, "embedding: token index out of vocab range");
            i as i32
        }).collect();
        let idx_dev = stream.clone_htod(&idx_i32).expect("cuda embedding: indices htod failed");

        let out_data = {
            let w_n = self.weight.borrow();
            let w_gpu = w_n.gpu.as_ref().expect("cuda embedding: weight not on GPU");
            let mut out = stream.alloc_zeros::<f32>(total).expect("cuda embedding: out alloc failed");
            let f = module().load_function("embed_forward").expect("embed_forward not found");
            let cfg = LaunchConfig::for_num_elems(total as u32);
            let (sl, dd) = (seq_len as i32, dim as i32);
            let mut b = stream.launch_builder(&f);
            b.arg(&mut out); b.arg(&w_gpu.data); b.arg(&idx_dev); b.arg(&sl); b.arg(&dd);
            unsafe { b.launch(cfg).expect("cuda embedding: forward launch failed"); }
            out
        };

        let out_node = Node::new(vec![], vec![seq_len, dim]);
        {
            let mut node = out_node.borrow_mut();
            node.parents = vec![self.weight.clone()];
            let grad = Rc::new(RefCell::new(stream.alloc_zeros::<f32>(total).expect("cuda embedding: grad alloc")));
            node.gpu = Some(GpuBuffers { data: out_data, grad: grad.clone() });

            let w_bwd = self.weight.clone();
            node.backward_fn = Some(Box::new(move |_grad: &Vec<f32>| {
                let stream = backend::stream();
                let og = grad.borrow();
                let w_n = w_bwd.borrow();
                let w_gpu = w_n.gpu.as_ref().expect("cuda embedding bwd: weight not on GPU");
                let mut w_grad = w_gpu.grad.borrow_mut();
                let f = module().load_function("embed_backward").expect("embed_backward not found");
                let cfg = LaunchConfig::for_num_elems(total as u32);
                let (sl, dd) = (seq_len as i32, dim as i32);
                let mut b = stream.launch_builder(&f);
                b.arg(&mut *w_grad); b.arg(&*og); b.arg(&idx_dev); b.arg(&sl); b.arg(&dd);
                unsafe { b.launch(cfg).expect("cuda embedding bwd: launch failed"); }
            }));
        }
        out_node
    }
}