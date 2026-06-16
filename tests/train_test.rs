use rstorch::tensor::Tensor;
use rstorch::nn::sequential::Sequential;
use rstorch::nn::linear::Linear;
use rstorch::nn::activations::relu::ReLU;
use rstorch::nn::loss::MSELoss;
use rstorch::data::dataloader::DataLoader;
use rstorch::train::train;
use rstorch::init::he;

#[test]
fn train_mlp_sum() {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..100 {
        let a = (i % 10) as f32 * 0.1;
        let b = ((i * 7) % 10) as f32 * 0.1;
        inputs.push(vec![a, b]);
        targets.push(vec![a + b]);
    }

    let mut loader = DataLoader::new(inputs, targets, 8);

    let mut model = Sequential {
        list: vec![
            Box::new(Linear {
                weights: Tensor::new(he::he(2, 8), vec![2, 8]),
                bias: Tensor::new(vec![0.0; 8], vec![8]),
                weights_node: None,
                bias_node: None,
            }),
            Box::new(ReLU {}),
            Box::new(Linear {
                weights: Tensor::new(he::he(8, 1), vec![8, 1]),
                bias: Tensor::new(vec![0.0], vec![1]),
                weights_node: None,
                bias_node: None,
            }),
        ],
    };

    let history = train(&mut model, &mut loader, &MSELoss, &[2], &[1], 0.05, 30);

    let first = history[0];
    let last = history[history.len() - 1];
    println!("first epoch loss: {}, last epoch loss: {}", first, last);

    assert!(last < first * 0.3, "loss nie spadl wystarczajaco: {} -> {}", first, last);
    println!("train mlp sum ok");
}