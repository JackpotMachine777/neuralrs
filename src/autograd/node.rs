use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

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

pub fn backward_graph(root: &Rc<RefCell<Node>>) {
    let mut topo : Vec<Rc<RefCell<Node>>> = Vec::new();
    let mut visited: HashSet<*const Node> = HashSet::new();

    build_topo(root, &mut visited, &mut topo);

    for node in topo.iter().rev() {
        let grad = node.borrow().grad.clone();
        
        if let Some(f) = &node.borrow().backward_fn {
            f(&grad);
        }
    }
}

fn build_topo(
    node: &Rc<RefCell<Node>>,
    visited: &mut HashSet<*const Node>,
    topo: &mut Vec<Rc<RefCell<Node>>>,
) {
    let ptr = Rc::as_ptr(node) as *const Node;
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let parents = node.borrow().parents.clone();
    for parent in &parents {
        build_topo(parent, visited, topo);
    }

    topo.push(node.clone());
}