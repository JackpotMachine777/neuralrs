use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::node::Node;

pub trait Module {
    fn forward(&mut self, input: Rc<RefCell<Node>>) -> Rc<RefCell<Node>>;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
    fn zero_grad(&mut self);
    fn sync_grads(&mut self) {}
    fn set_training(&mut self, _training: bool) {}
}