#![cfg(feature = "cuda")]

use neuralrs::autograd::node::{backward_graph, Node};
use neuralrs::autograd::graph as cpu;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::graph::add;

#[test]
fn cuda_resident_add_forward() {
    let n: usize = 5000;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.002).collect();
    let c: Vec<f32> = (0..n).map(|i| (i % 50) as f32 - 25.0).collect();

    let cpu: Vec<f32> = (0..n).map(|i| a[i] + b[i] + c[i]).collect();

    let ga = Node::new(a, vec![n]);
    let gb = Node::new(b, vec![n]);
    let gc = Node::new(c, vec![n]);
    
    gpu::to_cuda(&ga);
    gpu::to_cuda(&gb);
    gpu::to_cuda(&gc);

    let ab = add(&ga, &gb);
    let sum = add(&ab, &gc);
    let out = gpu::to_host(&sum);

    assert_eq!(out.len(), n);
    for i in 0..n {
        assert!(
            (out[i] - cpu[i]).abs() < 1e-4,
            "mismatch at {i}: gpu {} cpu {}",
            out[i],
            cpu[i]
        );
    }
    println!("resident (a+b)+c forward: gpu matches cpu over {n} elements");
}

#[test]
fn cuda_resident_add_backward() {
    let n: usize = 4096;
    let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.05 - 0.5).collect();
    let c: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
    let seed: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.3 + 0.1).collect();

    let ca = Node::new(a.clone(), vec![n]);
    let cb = Node::new(b.clone(), vec![n]);
    let cc = Node::new(c.clone(), vec![n]);
    let csum = cpu::add(cpu::add(ca.clone(), cb.clone()), cc.clone());

    csum.borrow_mut().grad = seed.clone();
    backward_graph(&csum);
    
    let cga = ca.borrow().grad.clone();
    let cgb = cb.borrow().grad.clone();
    let cgc = cc.borrow().grad.clone();

    let ga = Node::new(a, vec![n]);
    let gb = Node::new(b, vec![n]);
    let gc = Node::new(c, vec![n]);

    gpu::to_cuda(&ga);
    gpu::to_cuda(&gb);
    gpu::to_cuda(&gc);

    let gsum = add(&add(&ga, &gb), &gc);
    gpu::set_grad(&gsum, &seed);
    backward_graph(&gsum);

    let gga = gpu::read_grad(&ga);
    let ggb = gpu::read_grad(&gb);
    let ggc = gpu::read_grad(&gc);

    for i in 0..n {
        assert!((gga[i] - cga[i]).abs() < 1e-4, "a.grad at {i}: gpu {} cpu {}", gga[i], cga[i]);
        assert!((ggb[i] - cgb[i]).abs() < 1e-4, "b.grad at {i}: gpu {} cpu {}", ggb[i], cgb[i]);
        assert!((ggc[i] - cgc[i]).abs() < 1e-4, "c.grad at {i}: gpu {} cpu {}", ggc[i], cgc[i]);
    }

    println!("resident (a+b)+c backward: gpu grads match cpu over {n} elements");
}

#[test]
fn cuda_resident_bias_add_backward() {
    let (batch, features) = (32usize, 48usize);
    let x: Vec<f32> = (0..batch * features).map(|i| (i % 19) as f32 * 0.1 - 0.9).collect();
    let bias: Vec<f32> = (0..features).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
    let seed: Vec<f32> = (0..batch * features).map(|i| (i % 5) as f32 * 0.3 + 0.1).collect();

    let cx = Node::new(x.clone(), vec![batch, features]);
    let cb = Node::new(bias.clone(), vec![features]);
    let cout = cpu::add(cx.clone(), cb.clone());
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let cout_data = cout.borrow().data.clone();
    let cgx = cx.borrow().grad.clone();
    let cgb = cb.borrow().grad.clone();

    let gx = Node::new(x, vec![batch, features]);
    let gb = Node::new(bias, vec![features]);
    gpu::to_cuda(&gx);
    gpu::to_cuda(&gb);
    let gout = add(&gx, &gb);
    let gout_data = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let ggx = gpu::read_grad(&gx);
    let ggb = gpu::read_grad(&gb);

    for i in 0..batch * features {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-4, "fwd at {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
        assert!((ggx[i] - cgx[i]).abs() < 1e-4, "x.grad at {i}: gpu {} cpu {}", ggx[i], cgx[i]);
    }
    for f in 0..features {
        assert!((ggb[f] - cgb[f]).abs() < 1e-4, "bias.grad at {f}: gpu {} cpu {}", ggb[f], cgb[f]);
    }
    println!("resident bias-add forward+backward: gpu matches cpu (batch {batch}, features {features})");
}