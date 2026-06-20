use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

/// The shared interface every layer implements.
///
/// If a type is a `Module`, it can be dropped into a [`Sequential`] and treated
/// like any other layer. The required pieces:
/// - `forward` — push an input through and get an output (building the autograd
///   graph along the way)
/// - `parameters` — hand back the layer's trainable tensors so the optimizer can
///   update them
/// - `zero_grad` — clear out the gradients
///
/// `sync_grads` and `set_training` are optional (they default to doing nothing);
/// layers only override them if they need to — e.g. Dropout and BatchNorm behave
/// differently in training vs eval, so they override `set_training`.
///
/// [`Sequential`]: crate::nn::sequential::Sequential
pub trait Module {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>>;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
    fn zero_grad(&mut self);
    fn sync_grads(&mut self) {}
    fn set_training(&mut self, _training: bool) {}
}