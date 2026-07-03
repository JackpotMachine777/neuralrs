//! The autograd engine, the part that makes training possible.
//!
//! Every math operation you do builds a graph of [`Node`]s. Each node remembers
//! what it was made from (its `parents`) and how to hand its gradient back to
//! them (its `backward_fn`). Once you have a final value (like a loss), calling
//! backward walks that graph in reverse and fills in everyone's gradients.
//!
//! Key thing to keep straight: backward only **computes** gradients (it fills in
//! the `grad` buffers). It never touches the actual values, changing the
//! weights is a separate step done by the optimizer afterwards.

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// One node in the computation graph.
///
/// When you compute something like `c = a + b`, a new node `c` is created that
/// remembers it came from `a` and `b`. That's what makes gradients possible
/// later — the graph knows how everything was built.
///
/// Fields:
/// - `data` — the actual values (the result of whatever op made this node)
/// - `grad` — this node's gradient, starts at zero, gets filled during backward
/// - `shape` — how to read `data`
/// - `parents` — the nodes this one was built from (`[a, b]` for `a + b`)
/// - `backward_fn` — the recipe for how to split this node's gradient back to
///   its parents. For addition it passes the gradient straight through; for
///   multiplication it scales by the other operand; and so on. This is the
///   chain rule, one node at a time.
/// - `requires_grad` — whether gradients should flow *into* this node. Leaves
///   that are pure inputs (like an image batch) can set this to `false` so ops
///   skip computing a gradient nobody reads; currently honored by the CUDA
///   conv2d input gradient.
pub struct Node {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub parents: Vec<Rc<RefCell<Node>>>,
    pub backward_fn: Option<Box<dyn Fn(&Vec<f32>)>>,
    pub requires_grad: bool,
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuBuffers>,
}

#[cfg(feature = "cuda")]
pub struct GpuBuffers {
    pub data: CudaSlice<f32>,
    pub grad: Rc<RefCell<CudaSlice<f32>>>,
}

impl Node {
    /// Creates a fresh node from data and a shape, wrapped in `Rc<RefCell<...>>`
    /// so it can be shared around the graph.
    ///
    /// Starts with no parents and no backward function, those get set by
    /// whatever operation builds on top of it. Gradient starts at zero.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Rc<RefCell<Node>> {
        let len = data.len();

        Rc::new(RefCell::new(Node {
            data,
            grad: vec![0.0; len],
            shape,
            parents: vec![],
            backward_fn: None,
            requires_grad: true,
            #[cfg(feature = "cuda")]
            gpu: None,
        }))
    }

    /// Where this node's data currently lives.
    pub fn device(&self) -> crate::device::Device {
        #[cfg(feature = "cuda")]
        if self.gpu.is_some() { return crate::device::Device::CUDA(0); }
        
        crate::device::Device::CPU
    }

    /// Recursive backward pass: hand this node's gradient to its parents, then
    /// tell each parent to do the same.
    ///
    /// Works great for straight-line graphs. The catch: if a node is used in
    /// more than one place (like a residual connection that branches and
    /// re-joins), this can reach it before all its gradient paths have added
    /// their share, giving it an incomplete gradient. For those cases use
    /// [`backward_graph`], which orders things properly first.
    ///
    /// Remember: this fills in `grad`. It does **not** change `data`.
    pub fn backward(&mut self){
        if let Some(f) = &self.backward_fn { f(&self.grad); }

        for parent in &self.parents {
            parent.borrow_mut().backward();
        }
    }
}

/// Backward pass that handles any graph shape correctly, including branches.
///
/// First it puts all the nodes in topological order — meaning no node gets
/// processed until everything that uses it has already handed back its share of
/// the gradient. Then it walks that order in reverse, letting each node split
/// its (now complete) gradient back to its parents.
///
/// This is the one to use for anything with branching or residual connections
/// (attention, transformers). Like the recursive version, it only fills in
/// `grad` — it never changes the values.
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

/// Depth-first topological sort helper.
///
/// Visits all of a node's parents before adding the node itself to the list, so
/// the final order has parents before children. The `visited` set (keyed by raw
/// pointer) makes sure a shared node is only added once, important when the
/// graph branches and the same node is reachable through multiple paths.
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