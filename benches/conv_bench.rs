use criterion::{criterion_group, criterion_main, Criterion};
use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::conv::Conv2d;
use rstorch::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

fn conv_benchmark(c: &mut Criterion) {
    let c_in = 16;
    let c_out = 32;
    let in_h = 64;
    let in_w = 64;
    let kh = 3;
    let kw = 3;

    let w_len = c_out * c_in * kh * kw;
    let weight: Vec<f32> = (0..w_len).map(|i| (i % 7) as f32 * 0.01).collect();
    let bias = vec![0.0; c_out];

    let in_len = c_in * in_h * in_w;
    let input_data: Vec<f32> = (0..in_len).map(|i| (i % 5) as f32 * 0.1).collect();

    c.bench_function("conv 16->32 64x64 k3", |bench| {
        bench.iter(|| {
            let mut conv = Conv2d {
                weight: Tensor::new(weight.clone(), vec![c_out, c_in, kh, kw]),
                bias: Tensor::new(bias.clone(), vec![c_out]),
                c_in, c_out, kh, kw, stride: 1, in_h, in_w,
                weight_grad: Rc::new(RefCell::new(vec![0.0; w_len])),
                bias_grad: Rc::new(RefCell::new(vec![0.0; c_out])),
            };
            let input = Node::new(input_data.clone(), vec![c_in, in_h, in_w]);
            conv.forward(input)
        })
    });
}

criterion_group!(benches, conv_benchmark);
criterion_main!(benches);