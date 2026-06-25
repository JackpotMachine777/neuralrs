//! Cross-entropy loss on the GPU (operates on resident logits).
//!
//! Mirrors the CPU `CrossEntropyLoss`: takes raw logits (not softmaxed), does a
//! numerically-stable log-softmax internally, cross-entropy against a one-hot
//! target, averaged over the batch. Forward returns the scalar loss; backward
//! seeds the logits' gradient with the fused form `(softmax - target) / batch`,
//! after which `backward_graph` propagates through the resident ops.

use std::cell::RefCell;
use std::rc::Rc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::autograd::node::Node;
use crate::cuda::backend;

const KERNEL: &str = r#"
    extern "C" __global__ void ce_forward(float* row_loss, const float* logits, const float* target, int batch, int classes) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;

        if(b < batch) {
            int start = b * classes;
            float max = logits[start];

            for(int c = 1; c < classes; ++c) max = fmaxf(max, logits[start + c]);

            float sum_exp = 0.0f;
            for(int c = 0; c < classes; ++c) sum_exp += expf(logits[start + c] - max);
        
            float log_sum_exp = logf(sum_exp) + max;

            float loss = 0.0f;
            for(int c = 0; c < classes; ++c) loss += -target[start + c] * (logits[start + c] - log_sum_exp);
            
            row_loss[b] = loss;
        }
    }

    extern "C" __global__ void ce_backward(float* grad, const float* logits, const float* target, int batch, int classes) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;

        if(b < batch) {
            int start = b * classes;
            float max = logits[start];

            for(int c = 1; c < classes; ++c) max = fmaxf(max, logits[start + c]);
            
            float sum = 0.0f;
            for(int c = 0; c < classes; ++c) sum += expf(logits[start + c] - max);
            
            float inv_batch = 1.0f / (float)batch;
            for(int c = 0; c < classes; ++c){
                float softmax_c = expf(logits[start + c] - max) / sum;
                grad[start + c] = (softmax_c - target[start + c]) * inv_batch;
            }
        }
    }
"#;

crate::kernel_module!(KERNEL);

fn batch_classes(node: &Rc<RefCell<Node>>) -> (usize, usize) {
    let n = node.borrow();

    if n.shape.len() == 2 { (n.shape[0], n.shape[1]) }
    else { (1, n.shape[0]) }
}

/// Cross-entropy loss of resident logits against a resident one-hot target,
/// averaged over the batch. Both nodes must be on the GPU.
///
/// # Panics
/// If either node is not on the GPU.
pub fn cross_entropy(pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) -> f32 {
    let stream = backend::stream();
    let (batch, classes) = batch_classes(pred);

    let p = pred.borrow();
    let t = target.borrow();
    let logits = &p.gpu.as_ref().expect("cuda ce: logits not on GPU").data;
    let tgt = &t.gpu.as_ref().expect("cuda ce: target not on GPU").data;

    let mut row_loss = stream.alloc_zeros::<f32>(batch).expect("cuda ce: alloc failed");

    let f = module().load_function("ce_forward").expect("ce_forward not found");
    let cfg = LaunchConfig::for_num_elems(batch as u32);
    let (b, c) = (batch as i32, classes as i32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut row_loss);
    builder.arg(logits);
    builder.arg(tgt);
    builder.arg(&b);
    builder.arg(&c);
    unsafe { builder.launch(cfg).expect("cuda ce: forward launch failed"); }

    let host = stream.clone_dtoh(&row_loss).expect("cuda ce: dtoh failed");
    host.iter().sum::<f32>() / batch as f32
}

/// Seeds the logits' gradient with `(softmax - target) / batch` (the fused
/// softmax + cross-entropy gradient), writing into the resident grad buffer. Run
/// [`backward_graph`](crate::autograd::node::backward_graph) afterwards to
/// propagate through the rest of the graph.
///
/// # Panics
/// If either node is not on the GPU.
pub fn cross_entropy_backward(pred: &Rc<RefCell<Node>>, target: &Rc<RefCell<Node>>) {
    let stream = backend::stream();
    let (batch, classes) = batch_classes(pred);

    let p = pred.borrow();
    let t = target.borrow();
    let p_gpu = p.gpu.as_ref().expect("cuda ce: logits not on GPU");
    let logits = &p_gpu.data;
    let tgt = &t.gpu.as_ref().expect("cuda ce: target not on GPU").data;
    let mut grad = p_gpu.grad.borrow_mut();

    let f = module().load_function("ce_backward").expect("ce_backward not found");
    let cfg = LaunchConfig::for_num_elems(batch as u32);
    let (b, c) = (batch as i32, classes as i32);
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut *grad);
    builder.arg(logits);
    builder.arg(tgt);
    builder.arg(&b);
    builder.arg(&c);
    unsafe { builder.launch(cfg).expect("cuda ce: backward launch failed"); }
}