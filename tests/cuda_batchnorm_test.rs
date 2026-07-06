#![cfg(feature = "cuda")]

use neuralrs::autograd::node::Node;
use neuralrs::cuda::runtime as gpu;
use neuralrs::cuda::nn::batchnorm;
use neuralrs::nn::batchnorm::BatchNorm;
use neuralrs::nn::module::Module;
use neuralrs::tensor::Tensor;
use neuralrs::autograd::node::backward_graph;
use neuralrs::cuda::nn::batchnorm2d;
use neuralrs::nn::batchnorm2d::BatchNorm2d;
use std::rc::Rc;
use std::cell::RefCell;

fn cpu_bn(features: usize, training: bool, rm: Vec<f32>, rv: Vec<f32>) -> BatchNorm {
    BatchNorm {
        gamma: Tensor::new((0..features).map(|i| 0.5 + (i % 4) as f32 * 0.3).collect(), vec![features]),
        beta: Tensor::new((0..features).map(|i| (i % 3) as f32 * 0.2 - 0.2).collect(), vec![features]),
        epsilon: 1e-5,
        num_features: features,
        gamma_grad: Rc::new(RefCell::new(vec![0.0; features])),
        beta_grad: Rc::new(RefCell::new(vec![0.0; features])),
        running_mean: rm,
        running_var: rv,
        momentum: 0.9,
        training,
    }
}

fn cpu_bn2d(channels: usize, training: bool, rm: Vec<f32>, rv: Vec<f32>) -> BatchNorm2d {
    BatchNorm2d {
        gamma: Tensor::new((0..channels).map(|i| 0.5 + (i % 4) as f32 * 0.3).collect(), vec![channels]),
        beta: Tensor::new((0..channels).map(|i| (i % 3) as f32 * 0.2 - 0.2).collect(), vec![channels]),
        epsilon: 1e-5,
        num_channels: channels,
        gamma_grad: Rc::new(RefCell::new(vec![0.0; channels])),
        beta_grad: Rc::new(RefCell::new(vec![0.0; channels])),
        running_mean: rm,
        running_var: rv,
        momentum: 0.9,
        training,
    }
}

#[test]
fn cuda_batchnorm_forward_training() {
    let (batch, features) = (16usize, 8usize);
    let n = batch * features;
    let input: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 47) as f32 * 0.1 - 2.0).collect();

    let mut layer = cpu_bn(features, true, vec![0.0; features], vec![1.0; features]);
    let gamma_v = layer.gamma.storage.data.clone();
    let beta_v = layer.beta.storage.data.clone();
    let cin = Node::new(input.clone(), vec![batch, features]);
    let cout = layer.forward(cin);
    let cout_data = cout.borrow().data.clone();
    let c_rmean = layer.running_mean.clone();
    let c_rvar = layer.running_var.clone();

    let gi = Node::new(input, vec![batch, features]);
    let gg = Node::new(gamma_v, vec![features]);
    let gb = Node::new(beta_v, vec![features]);
    let rmean = Node::new(vec![0.0; features], vec![features]);
    let rvar = Node::new(vec![1.0; features], vec![features]);
    for x in [&gi, &gg, &gb, &rmean, &rvar] { gpu::to_cuda(x); }
    let gout = batchnorm(&gi, &gg, &gb, &rmean, &rvar, 0.9, 1e-5, true);
    let gout_data = gpu::to_host(&gout);
    let g_rmean = gpu::to_host(&rmean);
    let g_rvar = gpu::to_host(&rvar);

    for i in 0..n {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-4, "out at {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
    }
    for f in 0..features {
        assert!((g_rmean[f] - c_rmean[f]).abs() < 1e-5, "running_mean {f}");
        assert!((g_rvar[f] - c_rvar[f]).abs() < 1e-5, "running_var {f}");
    }
    println!("resident batchnorm forward (train): out + running stats match cpu");
}

#[test]
fn cuda_batchnorm_forward_eval() {
    let (batch, features) = (16usize, 8usize);
    let n = batch * features;
    let input: Vec<f32> = (0..n).map(|i| ((i * 9 + 1) % 31) as f32 * 0.1 - 1.5).collect();
    let rm: Vec<f32> = (0..features).map(|i| (i % 5) as f32 * 0.1).collect();
    let rv: Vec<f32> = (0..features).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();

    let mut layer = cpu_bn(features, false, rm.clone(), rv.clone());
    let gamma_v = layer.gamma.storage.data.clone();
    let beta_v = layer.beta.storage.data.clone();
    let cin = Node::new(input.clone(), vec![batch, features]);
    let cout = layer.forward(cin);
    let cout_data = cout.borrow().data.clone();

    let gi = Node::new(input, vec![batch, features]);
    let gg = Node::new(gamma_v, vec![features]);
    let gb = Node::new(beta_v, vec![features]);
    let rmean = Node::new(rm, vec![features]);
    let rvar = Node::new(rv, vec![features]);
    for x in [&gi, &gg, &gb, &rmean, &rvar] { gpu::to_cuda(x); }
    let gout = batchnorm(&gi, &gg, &gb, &rmean, &rvar, 0.9, 1e-5, false);
    let gout_data = gpu::to_host(&gout);

    for i in 0..n {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-4, "eval out at {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
    }
    println!("resident batchnorm forward (eval): out matches cpu");
}



#[test]
fn cuda_batchnorm_backward_training() {
    let (batch, features) = (16usize, 8usize);
    let n = batch * features;
    let input: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 47) as f32 * 0.1 - 2.0).collect();

    let mut layer = cpu_bn(features, true, vec![0.0; features], vec![1.0; features]);
    let gamma_v = layer.gamma.storage.data.clone();
    let beta_v = layer.beta.storage.data.clone();
    let cin = Node::new(input.clone(), vec![batch, features]);
    let cout = layer.forward(cin.clone());
    let out_len = cout.borrow().data.len();
    let seed: Vec<f32> = (0..out_len).map(|i| (i % 9) as f32 * 0.15 + 0.05).collect();
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let c_dgamma = layer.gamma_grad.borrow().clone();
    let c_dbeta = layer.beta_grad.borrow().clone();
    let c_dinput = cin.borrow().grad.clone();

    let gi = Node::new(input, vec![batch, features]);
    let gg = Node::new(gamma_v, vec![features]);
    let gb = Node::new(beta_v, vec![features]);
    let rmean = Node::new(vec![0.0; features], vec![features]);
    let rvar = Node::new(vec![1.0; features], vec![features]);
    for x in [&gi, &gg, &gb, &rmean, &rvar] { gpu::to_cuda(x); }
    let gout = batchnorm(&gi, &gg, &gb, &rmean, &rvar, 0.9, 1e-5, true);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let g_dgamma = gpu::read_grad(&gg);
    let g_dbeta = gpu::read_grad(&gb);
    let g_dinput = gpu::read_grad(&gi);

    for f in 0..features {
        assert!((g_dgamma[f] - c_dgamma[f]).abs() < 1e-3, "dgamma {f}: gpu {} cpu {}", g_dgamma[f], c_dgamma[f]);
        assert!((g_dbeta[f] - c_dbeta[f]).abs() < 1e-3, "dbeta {f}: gpu {} cpu {}", g_dbeta[f], c_dbeta[f]);
    }
    for i in 0..n {
        assert!((g_dinput[i] - c_dinput[i]).abs() < 1e-3, "dinput {i}: gpu {} cpu {}", g_dinput[i], c_dinput[i]);
    }
    println!("resident batchnorm backward (train): dgamma/dbeta/dinput match cpu");
}

#[test]
fn cuda_batchnorm2d_train() {
    let (n, c, h, w) = (4usize, 3usize, 5usize, 5usize);
    let total = n * c * h * w;
    let input: Vec<f32> = (0..total).map(|i| ((i * 13 + 5) % 47) as f32 * 0.1 - 2.0).collect();
    let seed: Vec<f32> = (0..total).map(|i| (i % 9) as f32 * 0.15 + 0.05).collect();

    let mut layer = cpu_bn2d(c, true, vec![0.0; c], vec![1.0; c]);
    let gamma_v = layer.gamma.storage.data.clone();
    let beta_v = layer.beta.storage.data.clone();
    let cin = Node::new(input.clone(), vec![n, c, h, w]);
    let cout = layer.forward(cin.clone());
    let cout_data = cout.borrow().data.clone();
    cout.borrow_mut().grad = seed.clone();
    backward_graph(&cout);
    let c_rmean = layer.running_mean.clone();
    let c_rvar = layer.running_var.clone();
    let c_dgamma = layer.gamma_grad.borrow().clone();
    let c_dbeta = layer.beta_grad.borrow().clone();
    let c_dinput = cin.borrow().grad.clone();

    let gi = Node::new(input, vec![n, c, h, w]);
    let gg = Node::new(gamma_v, vec![c]);
    let gb = Node::new(beta_v, vec![c]);
    let rmean = Node::new(vec![0.0; c], vec![c]);
    let rvar = Node::new(vec![1.0; c], vec![c]);
    for x in [&gi, &gg, &gb, &rmean, &rvar] { gpu::to_cuda(x); }
    let gout = batchnorm2d(&gi, &gg, &gb, &rmean, &rvar, 0.9, 1e-5, true);
    let gout_data = gpu::to_host(&gout);
    gpu::set_grad(&gout, &seed);
    backward_graph(&gout);
    let g_rmean = gpu::to_host(&rmean);
    let g_rvar = gpu::to_host(&rvar);
    let g_dgamma = gpu::read_grad(&gg);
    let g_dbeta = gpu::read_grad(&gb);
    let g_dinput = gpu::read_grad(&gi);

    for i in 0..total {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-4, "out {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
        assert!((g_dinput[i] - c_dinput[i]).abs() < 1e-3, "dinput {i}: gpu {} cpu {}", g_dinput[i], c_dinput[i]);
    }
    for ch in 0..c {
        assert!((g_rmean[ch] - c_rmean[ch]).abs() < 1e-5, "running_mean {ch}");
        assert!((g_rvar[ch] - c_rvar[ch]).abs() < 1e-5, "running_var {ch}");
        assert!((g_dgamma[ch] - c_dgamma[ch]).abs() < 1e-3, "dgamma {ch}: gpu {} cpu {}", g_dgamma[ch], c_dgamma[ch]);
        assert!((g_dbeta[ch] - c_dbeta[ch]).abs() < 1e-3, "dbeta {ch}: gpu {} cpu {}", g_dbeta[ch], c_dbeta[ch]);
    }
    println!("resident batchnorm2d (train): forward + running stats + backward match cpu");
}

#[test]
fn cuda_batchnorm2d_eval() {
    let (n, c, h, w) = (4usize, 3usize, 5usize, 5usize);
    let total = n * c * h * w;
    let input: Vec<f32> = (0..total).map(|i| ((i * 9 + 1) % 31) as f32 * 0.1 - 1.5).collect();
    let rm: Vec<f32> = (0..c).map(|i| (i % 5) as f32 * 0.1).collect();
    let rv: Vec<f32> = (0..c).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();

    let mut layer = cpu_bn2d(c, false, rm.clone(), rv.clone());
    let gamma_v = layer.gamma.storage.data.clone();
    let beta_v = layer.beta.storage.data.clone();
    let cin = Node::new(input.clone(), vec![n, c, h, w]);
    let cout = layer.forward(cin);
    let cout_data = cout.borrow().data.clone();

    let gi = Node::new(input, vec![n, c, h, w]);
    let gg = Node::new(gamma_v, vec![c]);
    let gb = Node::new(beta_v, vec![c]);
    let rmean = Node::new(rm, vec![c]);
    let rvar = Node::new(rv, vec![c]);
    for x in [&gi, &gg, &gb, &rmean, &rvar] { gpu::to_cuda(x); }
    let gout = batchnorm2d(&gi, &gg, &gb, &rmean, &rvar, 0.9, 1e-5, false);
    let gout_data = gpu::to_host(&gout);

    for i in 0..total {
        assert!((gout_data[i] - cout_data[i]).abs() < 1e-4, "eval out {i}: gpu {} cpu {}", gout_data[i], cout_data[i]);
    }
    println!("resident batchnorm2d (eval): out matches cpu");
}