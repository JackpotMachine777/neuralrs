use neuralrs::nn::module::Module;
use neuralrs::nn::dropout::Dropout;
use neuralrs::autograd::node::Node;

#[test]
fn dropout_test(){
    let mut layer = Dropout {
        probability: 0.5,
        mask: vec![],
        training: true,
    };

    let input = Node::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let output = layer.forward(input);

    println!("output: {:?}", output.borrow().data);
    assert_eq!(output.borrow().shape, vec![2, 3]);
}