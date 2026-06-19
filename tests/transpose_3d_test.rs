use rstorch::autograd::graph::transpose::transpose;
use rstorch::autograd::node::{Node, backward_graph};

#[test]
fn transpose_3d_forward() {
    let data = vec![
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,  10.0, 11.0, 12.0,
    ];
    let input = Node::new(data, vec![2, 2, 3]);
    let out = transpose(input);

    assert_eq!(out.borrow().shape, vec![2, 3, 2]);
    let d = out.borrow().data.clone();

    let expected_b0 = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let expected_b1 = vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0];

    println!("output: {:?}", d);
    for i in 0..6 {
        assert!((d[i] - expected_b0[i]).abs() < 1e-6, "batch 0 mismatch at {}", i);
        assert!((d[6 + i] - expected_b1[i]).abs() < 1e-6, "batch 1 mismatch at {}", i);
    }

    println!("transpose 3d forward ok");
}

#[test]
fn transpose_3d_backward() {
    let data = vec![
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,  10.0, 11.0, 12.0,
    ];
    let input = Node::new(data.clone(), vec![2, 2, 3]);
    let out = transpose(input.clone());

    let grad_inj: Vec<f32> = (0..out.borrow().data.len()).map(|j| j as f32 + 1.0).collect();
    out.borrow_mut().grad = grad_inj.clone();
    backward_graph(&out);

    let in_grad = input.borrow().grad.clone();
    println!("input grad: {:?}", in_grad);

    let rows = 2; let cols = 3;
    for b in 0..2 {
        for i in 0..rows {
            for j in 0..cols {
                let in_idx = b * rows * cols + i * cols + j;
                let out_idx = b * rows * cols + j * rows + i;
                assert!((in_grad[in_idx] - grad_inj[out_idx]).abs() < 1e-6,
                    "grad mismatch b{} i{} j{}", b, i, j);
            }
        }
    }

    println!("transpose 3d backward ok");
}