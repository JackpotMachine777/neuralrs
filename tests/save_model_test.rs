use rstorch::tensor::Tensor;
use rstorch::nn::module::Module;
use rstorch::nn::sequential::Sequential;
use rstorch::nn::linear::Linear;
use rstorch::nn::activations::relu::ReLU;
use rstorch::serialize::{save_model, load_model};
use rstorch::autograd::node::Node;

fn make_model(seed_offset: f32) -> Sequential {
    Sequential {
        list: vec![
            Box::new(Linear {
                weights: Tensor::new((0..6).map(|i| i as f32 * 0.1 + seed_offset).collect(), vec![2, 3]),
                bias: Tensor::new(vec![seed_offset; 3], vec![3]),
                weights_node: None,
                bias_node: None,
            }),
            Box::new(ReLU {}),
            Box::new(Linear {
                weights: Tensor::new((0..3).map(|i| i as f32 * 0.2 + seed_offset).collect(), vec![3, 1]),
                bias: Tensor::new(vec![seed_offset], vec![1]),
                weights_node: None,
                bias_node: None,
            }),
        ],
    }
}

#[test]
fn save_load_model_roundtrip() {
    let path = "/tmp/test_model.txt";

    let mut model_a = make_model(0.0);
    let input_data = vec![1.0, 2.0];

    let out_a = model_a.forward(Node::new(input_data.clone(), vec![1, 2]));
    let pred_a = out_a.borrow().data.clone();

    save_model(&mut model_a, path);

    let mut model_b = make_model(5.0);
    let out_b_before = model_b.forward(Node::new(input_data.clone(), vec![1, 2]));
    let pred_b_before = out_b_before.borrow().data.clone();

    println!("A: {:?}", pred_a);
    println!("B before load: {:?}", pred_b_before);
    assert_ne!(pred_a, pred_b_before, "modele should differ before load");

    load_model(&mut model_b, path);

    let out_b_after = model_b.forward(Node::new(input_data.clone(), vec![1, 2]));
    let pred_b_after = out_b_after.borrow().data.clone();

    println!("B after load: {:?}", pred_b_after);
    assert_eq!(pred_a, pred_b_after, "after load B has to = A");

    println!("save/load model roundtrip ok");
}