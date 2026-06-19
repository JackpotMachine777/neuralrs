use rstorch::optim::scheduler::{Scheduler, StepLR, ExponentialLR, CosineAnnealingLR, WarmupCosine};

#[test]
fn step_lr() {
    let s = StepLR { base_lr: 0.1, step_size: 10, gamma: 0.5 };

    assert!((s.get_lr(0) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(9) - 0.1).abs() < 1e-6);
    assert!((s.get_lr(10) - 0.05).abs() < 1e-6);
    assert!((s.get_lr(19) - 0.05).abs() < 1e-6);
    assert!((s.get_lr(20) - 0.025).abs() < 1e-6);

    println!("step_lr ok");
}

#[test]
fn exponential_lr() {
    let s = ExponentialLR { base_lr: 1.0, gamma: 0.9 };

    assert!((s.get_lr(0) - 1.0).abs() < 1e-6);
    assert!((s.get_lr(1) - 0.9).abs() < 1e-6);
    assert!((s.get_lr(2) - 0.81).abs() < 1e-5);
    assert!((s.get_lr(10) - 0.9_f32.powi(10)).abs() < 1e-5);

    println!("exponential_lr ok");
}

#[test]
fn cosine_annealing() {
    let s = CosineAnnealingLR { base_lr: 1.0, min_lr: 0.0, t_max: 10 };
    assert!((s.get_lr(0) - 1.0).abs() < 1e-5);
    assert!((s.get_lr(10) - 0.0).abs() < 1e-5);
    assert!((s.get_lr(5) - 0.5).abs() < 1e-5);
    for step in 0..10 {
        assert!(s.get_lr(step) >= s.get_lr(step + 1), "not decreasing at {}", step);
    }
    println!("cosine_annealing ok");
}

#[test]
fn warmup_cosine() {
    let s = WarmupCosine { base_lr: 1.0, min_lr: 0.0, warmup_steps: 5, t_max: 15 };

    assert!((s.get_lr(0) - 0.2).abs() < 1e-5);
    assert!((s.get_lr(4) - 1.0).abs() < 1e-5);
    assert!((s.get_lr(5) - 1.0).abs() < 1e-5);
    assert!((s.get_lr(15) - 0.0).abs() < 1e-4);

    for step in 0..4 {
        assert!(s.get_lr(step) < s.get_lr(step + 1), "warmup not increasing at {}", step);
    }

    for step in 5..15 {
        assert!(s.get_lr(step) >= s.get_lr(step + 1), "cosine not decreasing at {}", step);
    }

    println!("warmup_cosine ok");
}

#[test]
fn warmup_cosine_shape_print() {
    let s = WarmupCosine { base_lr: 0.001, min_lr: 0.00001, warmup_steps: 3, t_max: 20 };

    print!("warmup+cosine lr curve: ");

    for step in 0..20 {
        print!("{:.5} ", s.get_lr(step));
    }

    println!();
}