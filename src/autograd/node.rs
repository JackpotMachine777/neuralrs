use std::rc::Rc;
use std::cell::RefCell;

pub struct Node {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub parents: Vec<Rc<RefCell<Node>>>,
    pub backward_fn: Option<Box<dyn Fn(&Vec<f32>)>>,
}

impl Node {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Rc<RefCell<Node>> {
        let len = data.len();

        Rc::new(RefCell::new(Node {
            data,
            grad: vec![0.0; len],
            shape,
            parents: vec![],
            backward_fn: None,
        }))
    }

    pub fn backward(&mut self){
        if let Some(f) = &self.backward_fn { f(&self.grad); }

        for parent in &self.parents {
            parent.borrow_mut().backward();
        }
    }
}