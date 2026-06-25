#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::{add, matmul};

#[test]
fn cuda_resident_linear_chain() {
    let (batch, in_f, out_f) = (16usize, 24usize, 10usize);
    let x: Vec<f32> = (0..batch * in_f).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let w: Vec<f32> = (0..in_f * out_f).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
    let bias: Vec<f32> = (0..out_f).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
    let seed: Vec<f32> = (0..batch * out_f).map(|i| (i % 7) as f32 * 0.2 + 0.1).collect();

    let cx = Node::new(x.clone(), vec![batch, in_f]);
    let cw = Node::new(w.clone(), vec![in_f, out_f]);
    let cb = Node::new(bias.clone(), vec![out_f]);
    let cy = cpu::add(cpu::matmul(cx.clone(), cw.clone()), cb.clone());
    cy.borrow_mut().grad = seed.clone();
    backward_graph(&cy);
    let cy_data = cy.borrow().data.clone();
    let cgx = cx.borrow().grad.clone();
    let cgw = cw.borrow().grad.clone();
    let cgb = cb.borrow().grad.clone();

    let gx = Node::new(x, vec![batch, in_f]);
    let gw = Node::new(w, vec![in_f, out_f]);
    let gb = Node::new(bias, vec![out_f]);
    gpu::to_cuda(&gx);
    gpu::to_cuda(&gw);
    gpu::to_cuda(&gb);
    let gy = add(&matmul(&gx, &gw), &gb);
    let gy_data = gpu::to_host(&gy);
    gpu::set_grad(&gy, &seed);
    backward_graph(&gy);
    let ggx = gpu::read_grad(&gx);
    let ggw = gpu::read_grad(&gw);
    let ggb = gpu::read_grad(&gb);

    for i in 0..batch * out_f {
        assert!((gy_data[i] - cy_data[i]).abs() < 1e-3, "fwd at {i}: gpu {} cpu {}", gy_data[i], cy_data[i]);
    }
    for i in 0..batch * in_f {
        assert!((ggx[i] - cgx[i]).abs() < 1e-3, "x.grad at {i}: gpu {} cpu {}", ggx[i], cgx[i]);
    }
    for i in 0..in_f * out_f {
        assert!((ggw[i] - cgw[i]).abs() < 1e-3, "w.grad at {i}: gpu {} cpu {}", ggw[i], cgw[i]);
    }
    for i in 0..out_f {
        assert!((ggb[i] - cgb[i]).abs() < 1e-3, "bias.grad at {i}: gpu {} cpu {}", ggb[i], cgb[i]);
    }
    println!("resident Linear chain (x@w + bias) forward+backward: gpu matches cpu ({batch}x{in_f} @ {in_f}x{out_f})");
}