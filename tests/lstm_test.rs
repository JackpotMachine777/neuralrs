use rstorch::tensor::Tensor;
use rstorch::nn::lstm::LSTMCell;
use rstorch::autograd::node::Node;
use rstorch::init::xavier;

fn make_cell(input_size: usize, hidden_size: usize) -> LSTMCell {
    let w = |a, b| Tensor::new(xavier::xavier(a, b), vec![a, b]);
    let bias = |h| Tensor::new(vec![0.0; h], vec![h]);
    LSTMCell {
        w_f: w(input_size, hidden_size), u_f: w(hidden_size, hidden_size), b_f: bias(hidden_size),
        w_i: w(input_size, hidden_size), u_i: w(hidden_size, hidden_size), b_i: bias(hidden_size),
        w_o: w(input_size, hidden_size), u_o: w(hidden_size, hidden_size), b_o: bias(hidden_size),
        w_g: w(input_size, hidden_size), u_g: w(hidden_size, hidden_size), b_g: bias(hidden_size),
        input_size, hidden_size,
        nodes: None,
    }
}

#[test]
fn lstm_sequence_bptt() {
    let input_size = 3;
    let hidden_size = 4;
    let mut cell = make_cell(input_size, hidden_size);
    cell.zero_grad();
 
    let seq = vec![
        Node::new(vec![1.0, 0.5, -0.2], vec![1, 3]),
        Node::new(vec![0.3, -0.1, 0.8], vec![1, 3]),
        Node::new(vec![-0.5, 0.2, 0.4], vec![1, 3]),
        Node::new(vec![0.1, 0.6, -0.3], vec![1, 3]),
    ];

    let mut h = Node::new(vec![0.0; hidden_size], vec![1, hidden_size]);
    let mut c = Node::new(vec![0.0; hidden_size], vec![1, hidden_size]);

    for x in &seq {
        let (h_new, c_new) = cell.step(x.clone(), h.clone(), c.clone());
        h = h_new;
        c = c_new;
    }

    println!("final h: {:?}", h.borrow().data);
    assert_eq!(h.borrow().shape, vec![1, hidden_size]);

    h.borrow_mut().grad = vec![1.0; hidden_size];
    h.borrow_mut().backward();

    cell.sync_grads();

    let uf_sum: f32 = cell.u_f.grad.iter().map(|x| x.abs()).sum();
    let ui_sum: f32 = cell.u_i.grad.iter().map(|x| x.abs()).sum();
    let uo_sum: f32 = cell.u_o.grad.iter().map(|x| x.abs()).sum();
    let ug_sum: f32 = cell.u_g.grad.iter().map(|x| x.abs()).sum();

    println!("u_f: {}, u_i: {}, u_o: {}, u_g: {}", uf_sum, ui_sum, uo_sum, ug_sum);

    assert!(uf_sum > 0.0, "forget gate recurrent grad zero!");
    assert!(ui_sum > 0.0, "input gate recurrent grad zero!");
    assert!(uo_sum > 0.0, "output gate recurrent grad zero!");
    assert!(ug_sum > 0.0, "candidate recurrent grad zero!");

    println!("lstm BPTT ok");
}