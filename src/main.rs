use rstorch::tensor::Tensor;
use rstorch::ops::matmul::matmul;

fn main() {
    let a = Tensor::new(
        vec![
            1.0, 2.0,
            3.0, 4.0
        ],
        vec![2, 2]
    );

    let b = Tensor::new(
        vec![
            5.0, 6.0,
            7.0, 8.0
        ],
        vec![2, 2]
    );

    let c = matmul(&a, &b);

    println!("Result: {:?}", c.data);
}