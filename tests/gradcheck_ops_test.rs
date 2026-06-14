use rstorch::autograd::node::Node;
use rstorch::autograd::graph::{sub, div, exp, pow};
use rstorch::autograd::engine;

fn num_grad_unary<F>(input: &Vec<f32>, f: F) -> Vec<f32> where F: Fn(&Vec<f32>) -> Vec<f32> {
    let h = 1e-3;
    let mut g = vec![0.0; input.len()];
    for i in 0..input.len() {
        let mut p = input.clone(); p[i] += h;
        let mut m = input.clone(); m[i] -= h;
        let lp: f32 = f(&p).iter().sum();
        let lm: f32 = f(&m).iter().sum();
        g[i] = (lp - lm) / (2.0 * h);
    }
    g
}

#[test]
fn gradcheck_sub() {
    let a = Node::new(vec![5.0, 3.0, 8.0], vec![3]);
    let b = Node::new(vec![2.0, 1.0, 4.0], vec![3]);
    let c = sub::sub(a.clone(), b.clone());
    engine::backward(c);
    assert_eq!(a.borrow().grad, vec![1.0, 1.0, 1.0]);
    assert_eq!(b.borrow().grad, vec![-1.0, -1.0, -1.0]);
    println!("sub ok");
}

#[test]
fn gradcheck_div() {
    let a_data = vec![6.0, 8.0, 10.0];
    let b_data = vec![2.0, 4.0, 5.0];

    let a = Node::new(a_data.clone(), vec![3]);
    let b = Node::new(b_data.clone(), vec![3]);
    let c = div::div(a.clone(), b.clone());
    engine::backward(c);
    let ga = a.borrow().grad.clone();

    let bd = b_data.clone();
    let num_a = num_grad_unary(&a_data, |ain| {
        ain.iter().zip(bd.iter()).map(|(x, y)| x / y).collect()
    });

    println!("div analytic da: {:?}", ga);
    println!("div numeric  da: {:?}", num_a);
    for i in 0..3 {
        assert!((ga[i] - num_a[i]).abs() < 1e-2, "div da mismatch at {}", i);
    }
    println!("div ok");
}

#[test]
fn gradcheck_exp() {
    let a_data = vec![0.5, 1.0, -1.0];
    let a = Node::new(a_data.clone(), vec![3]);
    let c = exp::exp(a.clone());
    engine::backward(c);
    let ga = a.borrow().grad.clone();

    let num = num_grad_unary(&a_data, |ain| ain.iter().map(|x| x.exp()).collect());

    println!("exp analytic: {:?}", ga);
    println!("exp numeric:  {:?}", num);
    for i in 0..3 {
        assert!((ga[i] - num[i]).abs() < 1e-2, "exp mismatch at {}", i);
    }
    println!("exp ok");
}

#[test]
fn gradcheck_pow() {
    let a_data = vec![2.0, 3.0, 4.0];
    let p = 3.0;
    let a = Node::new(a_data.clone(), vec![3]);
    let c = pow::pow(a.clone(), p);
    engine::backward(c);
    let ga = a.borrow().grad.clone();

    let num = num_grad_unary(&a_data, |ain| ain.iter().map(|x| x.powf(p)).collect());

    println!("pow analytic: {:?}", ga);
    println!("pow numeric:  {:?}", num);
    for i in 0..3 {
        assert!((ga[i] - num[i]).abs() < 1e-2, "pow mismatch at {}", i);
    }
    println!("pow ok");
}

#[test]
fn gradcheck_log() {
    use rstorch::autograd::graph::log;
    let a_data = vec![1.0, 2.0, 5.0];
    let a = Node::new(a_data.clone(), vec![3]);
    let c = log::log(a.clone());
    engine::backward(c);
    let ga = a.borrow().grad.clone();

    let num = num_grad_unary(&a_data, |ain| ain.iter().map(|x| x.ln()).collect());

    println!("log analytic: {:?}", ga);
    println!("log numeric:  {:?}", num);
    for i in 0..3 {
        assert!((ga[i] - num[i]).abs() < 1e-2, "log mismatch at {}", i);
    }
    println!("log ok");
}

#[test]
fn gradcheck_sqrt() {
    use rstorch::autograd::graph::sqrt;
    let a_data = vec![1.0, 4.0, 9.0];
    let a = Node::new(a_data.clone(), vec![3]);
    let c = sqrt::sqrt(a.clone());
    engine::backward(c);
    let ga = a.borrow().grad.clone();

    let num = num_grad_unary(&a_data, |ain| ain.iter().map(|x| x.sqrt()).collect());

    println!("sqrt analytic: {:?}", ga);
    println!("sqrt numeric:  {:?}", num);
    for i in 0..3 {
        assert!((ga[i] - num[i]).abs() < 1e-2, "sqrt mismatch at {}", i);
    }
    println!("sqrt ok");
}

#[test]
fn gradcheck_abs() {
    use rstorch::autograd::graph::abs;
    let a_data = vec![2.0, -3.0, 5.0];
    let a = Node::new(a_data.clone(), vec![3]);
    let c = abs::abs(a.clone());
    engine::backward(c);
    let ga = a.borrow().grad.clone();

    let num = num_grad_unary(&a_data, |ain| ain.iter().map(|x| x.abs()).collect());

    println!("abs analytic: {:?}", ga);
    println!("abs numeric:  {:?}", num);
    for i in 0..3 {
        assert!((ga[i] - num[i]).abs() < 1e-2, "abs mismatch at {}", i);
    }
    println!("abs ok");
}