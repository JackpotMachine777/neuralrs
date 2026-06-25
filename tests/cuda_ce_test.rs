#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::{add, matmul};
use neuralrs::cuda::loss::{cross_entropy, cross_entropy_backward};
use neuralrs::nn::loss::{CrossEntropyLoss, Loss};

#[test]
fn cuda_cross_entropy_through_linear() {
    let (batch, in_f, classes) = (16usize, 24usize, 10usize);
    let x: Vec<f32> = (0..batch * in_f).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let w: Vec<f32> = (0..in_f * classes).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
    let bias: Vec<f32> = (0..classes).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
    let mut target = vec![0.0f32; batch * classes];
    for b in 0..batch {
        target[b * classes + (b % classes)] = 1.0;
    }

    let cx = Node::new(x.clone(), vec![batch, in_f]);
    let cw = Node::new(w.clone(), vec![in_f, classes]);
    let cb = Node::new(bias.clone(), vec![classes]);
    let ct = Node::new(target.clone(), vec![batch, classes]);
    let clogits = cpu::add(cpu::matmul(cx.clone(), cw.clone()), cb.clone());
    let cpu_loss = CrossEntropyLoss.forward(&clogits, &ct);
    CrossEntropyLoss.backward(&clogits, &ct);
    let cgx = cx.borrow().grad.clone();
    let cgw = cw.borrow().grad.clone();
    let cgb = cb.borrow().grad.clone();

    let gx = Node::new(x, vec![batch, in_f]);
    let gw = Node::new(w, vec![in_f, classes]);
    let gb = Node::new(bias, vec![classes]);
    let gt = Node::new(target, vec![batch, classes]);
    gpu::to_cuda(&gx);
    gpu::to_cuda(&gw);
    gpu::to_cuda(&gb);
    gpu::to_cuda(&gt);
    let glogits = add(&matmul(&gx, &gw), &gb);
    let gpu_loss = cross_entropy(&glogits, &gt);
    cross_entropy_backward(&glogits, &gt);
    backward_graph(&glogits);
    let ggx = gpu::read_grad(&gx);
    let ggw = gpu::read_grad(&gw);
    let ggb = gpu::read_grad(&gb);

    assert!((gpu_loss - cpu_loss).abs() < 1e-3, "loss: gpu {gpu_loss} cpu {cpu_loss}");
    for i in 0..batch * in_f {
        assert!((ggx[i] - cgx[i]).abs() < 1e-3, "x.grad at {i}: gpu {} cpu {}", ggx[i], cgx[i]);
    }
    for i in 0..in_f * classes {
        assert!((ggw[i] - cgw[i]).abs() < 1e-3, "w.grad at {i}: gpu {} cpu {}", ggw[i], cgw[i]);
    }
    for i in 0..classes {
        assert!((ggb[i] - cgb[i]).abs() < 1e-3, "bias.grad at {i}: gpu {} cpu {}", ggb[i], cgb[i]);
    }
    println!("resident CE through Linear: loss + grads match cpu (loss={gpu_loss:.4}, batch {batch}, classes {classes})");
}