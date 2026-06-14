use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::conv::Conv2d;
use rstorch::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;
use std::time::Instant;

#[test]
fn bench_conv() {
    // realniejszy rozmiar: 16 kanalow wej, 32 filtry, obraz 64x64, filtr 3x3
    let c_in = 16;
    let c_out = 32;
    let in_h = 64;
    let in_w = 64;
    let kh = 3;
    let kw = 3;

    let w_len = c_out * c_in * kh * kw;
    let weight: Vec<f32> = (0..w_len).map(|i| (i % 7) as f32 * 0.01).collect();
    let bias = vec![0.0; c_out];

    let mut conv = Conv2d {
        weight: Tensor::new(weight, vec![c_out, c_in, kh, kw]),
        bias: Tensor::new(bias, vec![c_out]),
        c_in, c_out, kh, kw, stride: 1, in_h, in_w,
        weight_grad: Rc::new(RefCell::new(vec![0.0; w_len])),
        bias_grad: Rc::new(RefCell::new(vec![0.0; c_out])),
    };

    let in_len = c_in * in_h * in_w;
    let input_data: Vec<f32> = (0..in_len).map(|i| (i % 5) as f32 * 0.1).collect();
    let input = Node::new(input_data, vec![c_in, in_h, in_w]);

    let start = Instant::now();
    let out = conv.forward(input);
    let elapsed = start.elapsed();

    println!("checksum: {}", out.borrow().data[0]);
    println!("conv forward took: {:?}", elapsed);
}