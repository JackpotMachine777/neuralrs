use rstorch::tensor::Tensor;
use rstorch::nn::linear::Linear;
use rstorch::optim::clip::clip_grad_norm;

fn linear_with_grad(grad: Vec<f32>) -> Linear {
    let n = grad.len();
    let mut layer = Linear {
        weights: Tensor::new(vec![0.0; n], vec![1, n]),
        bias: Tensor::new(vec![0.0], vec![1]),
        weights_node: None,
        bias_node: None,
    };

    layer.weights.grad = grad;
    layer.bias.grad = vec![0.0];
    layer
}

#[test]
fn clip_scales_down_large_grad() {
    let mut layer = linear_with_grad(vec![3.0, 4.0]);

    clip_grad_norm(&mut layer, 1.0);

    let g = &layer.weights.grad;
    println!("clipped grad: {:?}", g);
    assert!((g[0] - 0.6).abs() < 1e-5);
    assert!((g[1] - 0.8).abs() < 1e-5);

    let new_norm = (g[0]*g[0] + g[1]*g[1]).sqrt();
    println!("new norm: {}", new_norm);
    assert!((new_norm - 1.0).abs() < 1e-5);

    println!("clip large ok");
}

#[test]
fn clip_leaves_small_grad() {
    let mut layer = linear_with_grad(vec![0.3, 0.4]);

    clip_grad_norm(&mut layer, 1.0);

    let g = &layer.weights.grad;
    println!("grad: {:?}", g);
    assert!((g[0] - 0.3).abs() < 1e-5);
    assert!((g[1] - 0.4).abs() < 1e-5);

    println!("clip small ok");
}