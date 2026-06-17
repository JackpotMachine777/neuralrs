use crate::tensor::Tensor;
use crate::autograd::node::Node;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub weight_node: Option<Rc<RefCell<Node>>>,
}

impl Embedding {
    pub fn forward(&mut self, indices: &[usize]) -> Rc<RefCell<Node>> {
        let seq_len = indices.len();
        let dim = self.embedding_dim;

        let mut data = vec![0.0; seq_len * dim];
        for (pos, &idx) in indices.iter().enumerate() {
            assert!(idx < self.vocab_size, "token index out of vocab range");

            for d in 0..dim {
                data[pos * dim + d] = self.weight.storage.data[idx * dim + d];
            }
        }

        let result = Node::new(data, vec![seq_len, dim]);

        if self.weight_node.is_none() {
            self.weight_node = Some(Node::new(
                self.weight.storage.data.clone(),
                self.weight.shape.clone(),
            ));
        }
        let w_node = self.weight_node.clone().unwrap();

        {
            let mut node = result.borrow_mut();
            node.parents = vec![w_node.clone()];
        }

        let indices_owned: Vec<usize> = indices.to_vec();
        result.borrow_mut().backward_fn = Some(Box::new(move |grad: &Vec<f32>| {
            let mut w = w_node.borrow_mut();

            for (pos, &idx) in indices_owned.iter().enumerate() {
                for d in 0..dim {
                    w.grad[idx * dim + d] += grad[pos * dim + d];
                }
            }
        }));

        result
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> { vec![&mut self.weight] }

    pub fn zero_grad(&mut self) {
        self.weight.grad = vec![0.0; self.weight.storage.data.len()];
        self.weight_node = None;
    }

    pub fn sync_grads(&mut self) {
        if let Some(w) = &self.weight_node {
            self.weight.grad = w.borrow().grad.clone();
        }
    }
}