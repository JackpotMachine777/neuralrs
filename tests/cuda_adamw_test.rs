#![cfg(feature = "cuda")]

use neuralrs::autograd::node::Node;
use neuralrs::cuda::optim::AdamW;
use neuralrs::cuda::runtime as gpu;
use neuralrs::optim::adamw::ADAMW;
use neuralrs::tensor::Tensor;

#[test]
fn cuda_adamw_matches_cpu() {
    let n = 256usize;
    let init: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
    let grad: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.05 - 0.25).collect();
    let (lr, b1, b2, eps, wd) = (0.001f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32);
    let steps = 10;

    let mut ct = Tensor::new(init.clone(), vec![n]);
    let mut copt = ADAMW { lr, beta1: b1, beta2: b2, epsilon: eps, weight_decay: wd, t: 0, m: vec![], v: vec![] };
    for _ in 0..steps {
        ct.grad = grad.clone();
        copt.step_params(&mut vec![&mut ct]);
    }
    let cpu_param = ct.storage.data.clone();

    let gp = Node::new(init, vec![n]);
    gpu::to_cuda(&gp);
    let mut gopt = AdamW::new(lr, b1, b2, eps, wd);
    for _ in 0..steps {
        gpu::set_grad(&gp, &grad);
        gopt.step(std::slice::from_ref(&gp));
    }
    let gpu_param = gpu::to_host(&gp);

    for i in 0..n {
        assert!((gpu_param[i] - cpu_param[i]).abs() < 1e-4, "param at {i}: gpu {} cpu {}", gpu_param[i], cpu_param[i]);
    }
    println!("resident AdamW matches cpu after {steps} steps over {n} params");
}

#[test]
fn cuda_adamw_state_roundtrip() {
    let n = 64usize;
    let init: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let g1: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.05 + 0.01).collect();
    let g2: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.04 - 0.08).collect();
    let g3: Vec<f32> = (0..n).map(|i| (i % 9) as f32 * 0.03 + 0.02).collect();

    // Control: one optimizer runs all three steps uninterrupted.
    let a = Node::new(init.clone(), vec![n]);
    gpu::to_cuda(&a);
    let mut oa = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);
    for g in [&g1, &g2] {
        gpu::set_grad(&a, g);
        oa.step(std::slice::from_ref(&a));
    }

    // Roundtrip: two steps, export, fresh optimizer, import.
    let b = Node::new(init, vec![n]);
    gpu::to_cuda(&b);
    let mut ob = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);
    for g in [&g1, &g2] {
        gpu::set_grad(&b, g);
        ob.step(std::slice::from_ref(&b));
    }
    let (t, moments) = ob.export_state();
    assert_eq!(t, 2);
    assert_eq!(moments.len(), 2);
    let mut ob2 = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);
    ob2.import_state(t, &moments);

    // One more identical step on both sides.
    gpu::set_grad(&a, &g3);
    oa.step(std::slice::from_ref(&a));
    gpu::set_grad(&b, &g3);
    ob2.step(std::slice::from_ref(&b));

    let pa = gpu::to_host(&a);
    let pb = gpu::to_host(&b);
    for i in 0..n {
        assert!((pa[i] - pb[i]).abs() < 1e-6, "param {i}: control {} vs roundtrip {}", pa[i], pb[i]);
    }
    println!("adamw state roundtrip: post-restore step matches an uninterrupted run");
}