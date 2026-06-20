use crate::nn::module::Module;
use crate::nn::loss::Loss;
use crate::data::dataloader::DataLoader;
use crate::autograd::node::Node;

/// A ready-made training loop for simple cases.
///
/// Runs `epochs` passes over the data loader: for each batch it does forward,
/// computes the loss, backpropagates, syncs gradients, and steps a plain SGD
/// update at the given `lr`. Returns the loss history. For anything fancier
/// (custom optimizers, schedulers) you'd write the loop yourself — see the MNIST
/// examples.
pub fn train<M: Module, L: Loss>(
    model: &mut M,
    loader: &mut DataLoader,
    loss_fn: &L,
    input_shape_tail: &[usize],
    target_shape_tail: &[usize],
    lr: f32,
    epochs: usize,
) -> Vec<f32> {
    let mut history = Vec::new();

    for epoch in 0..epochs {
        loader.shuffle();
        let mut epoch_loss = 0.0;
        let nb = loader.num_batches();

        for b in 0..nb {
            let (in_data, tgt_data, bs) = loader.get_batch(b);

            let mut in_shape = vec![bs];
            in_shape.extend_from_slice(input_shape_tail);

            let mut tgt_shape = vec![bs];
            tgt_shape.extend_from_slice(target_shape_tail);

            let input = Node::new(in_data, in_shape);
            let target = Node::new(tgt_data, tgt_shape);

            model.zero_grad();
            let output = model.forward(input);

            let loss = loss_fn.forward(&output, &target);
            epoch_loss += loss;

            loss_fn.backward(&output, &target);
            model.sync_grads();

            for p in model.parameters() {
                for i in 0..p.storage.data.len() {
                    p.storage.data[i] -= lr * p.grad[i];
                }
            }
        }

        let avg = epoch_loss / nb as f32;
        history.push(avg);
        println!("Epoch {epoch}: avg loss = {avg}");
    }

    history
}