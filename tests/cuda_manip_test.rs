#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::{bmm, concat_cols, slice_cols, softmax, transpose};
use std::cell::RefCell;
use std::rc::Rc;

type N = Rc<RefCell<Node>>;

fn run_cpu_unary(input: &[f32], shape: &[usize], seed: &[f32], op: impl Fn(N) -> N) -> (Vec<f32>, Vec<f32>) {
    let x = Node::new(input.to_vec(), shape.to_vec());
    let out = op(x.clone());
    out.borrow_mut().grad = seed.to_vec();
    backward_graph(&out);
    (out.borrow().data.clone(), x.borrow().grad.clone())
}

fn run_gpu_unary(input: &[f32], shape: &[usize], seed: &[f32], op: impl Fn(&N) -> N) -> (Vec<f32>, Vec<f32>) {
    let x = Node::new(input.to_vec(), shape.to_vec());
    gpu::to_cuda(&x);
    let out = op(&x);
    let fwd = gpu::to_host(&out);
    gpu::set_grad(&out, seed);
    backward_graph(&out);
    (fwd, gpu::read_grad(&x))
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch {} vs {}", a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() < tol, "{what} [{i}]: gpu {} cpu {}", a[i], b[i]);
    }
}

#[test]
fn cuda_transpose() {
    let (b, r, c) = (2usize, 3usize, 4usize);
    let input: Vec<f32> = (0..b * r * c).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let seed: Vec<f32> = (0..b * r * c).map(|i| (i % 7) as f32 * 0.1 + 0.2).collect();
    let (cf, cg) = run_cpu_unary(&input, &[b, r, c], &seed, |x| cpu::transpose(x));
    let (gf, gg) = run_gpu_unary(&input, &[b, r, c], &seed, |x| transpose(x));
    close(&gf, &cf, 1e-5, "transpose fwd");
    close(&gg, &cg, 1e-5, "transpose grad");
    println!("transpose: gpu matches cpu");
}

#[test]
fn cuda_slice_cols() {
    let (rows, total) = (4usize, 6usize);
    let (cs, ce) = (2usize, 5usize);
    let sw = ce - cs;
    let input: Vec<f32> = (0..rows * total).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let seed: Vec<f32> = (0..rows * sw).map(|i| (i % 5) as f32 * 0.1 + 0.3).collect();
    let (cf, cg) = run_cpu_unary(&input, &[rows, total], &seed, |x| cpu::slice_cols(x, cs, ce));
    let (gf, gg) = run_gpu_unary(&input, &[rows, total], &seed, |x| slice_cols(x, cs, ce));
    close(&gf, &cf, 1e-5, "slice_cols fwd");
    close(&gg, &cg, 1e-5, "slice_cols grad");
    println!("slice_cols: gpu matches cpu");
}

#[test]
fn cuda_softmax() {
    let (rows, width) = (4usize, 5usize);
    let input: Vec<f32> = (0..rows * width).map(|i| (i % 9) as f32 * 0.3 - 1.0).collect();
    let seed: Vec<f32> = (0..rows * width).map(|i| (i % 7) as f32 * 0.1 + 0.2).collect();
    let (cf, cg) = run_cpu_unary(&input, &[rows, width], &seed, |x| cpu::softmax(x));
    let (gf, gg) = run_gpu_unary(&input, &[rows, width], &seed, |x| softmax(x));
    close(&gf, &cf, 1e-4, "softmax fwd");
    close(&gg, &cg, 1e-4, "softmax grad");
    println!("softmax: gpu matches cpu");
}

#[test]
fn cuda_bmm() {
    let (batch, m, k, n) = (2usize, 3usize, 4usize, 5usize);
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let bb: Vec<f32> = (0..batch * k * n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let seed: Vec<f32> = (0..batch * m * n).map(|i| (i % 7) as f32 * 0.1 + 0.2).collect();

    let ca = Node::new(a.clone(), vec![batch, m, k]);
    let cbn = Node::new(bb.clone(), vec![batch, k, n]);
    let cout = cpu::bmm(ca.clone(), cbn.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let (cf, cga, cgb) = (cout.borrow().data.clone(), ca.borrow().grad.clone(), cbn.borrow().grad.clone());

    let ga = Node::new(a, vec![batch, m, k]);
    let gbn = Node::new(bb, vec![batch, k, n]);
    gpu::to_cuda(&ga);
    gpu::to_cuda(&gbn);
    let gout = bmm(&ga, &gbn);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gga, ggb) = (gpu::read_grad(&ga), gpu::read_grad(&gbn));

    close(&gf, &cf, 1e-4, "bmm fwd");
    close(&gga, &cga, 1e-4, "bmm grad_a");
    close(&ggb, &cgb, 1e-4, "bmm grad_b");
    println!("bmm: gpu matches cpu");
}

#[test]
fn cuda_concat_cols() {
    let rows = 4usize;
    let w = [2usize, 3usize, 1usize];
    let total: usize = w.iter().sum();
    let p0: Vec<f32> = (0..rows * w[0]).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let p1: Vec<f32> = (0..rows * w[1]).map(|i| (i % 9) as f32 * 0.1 - 0.4).collect();
    let p2: Vec<f32> = (0..rows * w[2]).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
    let seed: Vec<f32> = (0..rows * total).map(|i| (i % 6) as f32 * 0.1 + 0.2).collect();

    let c0 = Node::new(p0.clone(), vec![rows, w[0]]);
    let c1 = Node::new(p1.clone(), vec![rows, w[1]]);
    let c2 = Node::new(p2.clone(), vec![rows, w[2]]);
    let cout = cpu::concat_cols(vec![c0.clone(), c1.clone(), c2.clone()]);
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cf = cout.borrow().data.clone();
    let (cg0, cg1, cg2) = (c0.borrow().grad.clone(), c1.borrow().grad.clone(), c2.borrow().grad.clone());

    let g0 = Node::new(p0, vec![rows, w[0]]);
    let g1 = Node::new(p1, vec![rows, w[1]]);
    let g2 = Node::new(p2, vec![rows, w[2]]);
    gpu::to_cuda(&g0);
    gpu::to_cuda(&g1);
    gpu::to_cuda(&g2);
    let gout = concat_cols(&[g0.clone(), g1.clone(), g2.clone()]);
    let gf = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let (gg0, gg1, gg2) = (gpu::read_grad(&g0), gpu::read_grad(&g1), gpu::read_grad(&g2));

    close(&gf, &cf, 1e-5, "concat fwd");
    close(&gg0, &cg0, 1e-5, "concat grad0");
    close(&gg1, &cg1, 1e-5, "concat grad1");
    close(&gg2, &cg2, 1e-5, "concat grad2");
    println!("concat_cols: gpu matches cpu");
}