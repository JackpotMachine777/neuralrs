use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// He initialization — random weights scaled for ReLU-style activations.
///
/// Draws from a normal distribution scaled by `sqrt(2 / fan_in)`, which keeps
/// the signal variance roughly stable through ReLU layers. Returns a flat vector
/// of `fan_in * fan_out` values.
pub fn he(fan_in: usize, fan_out: usize) -> Vec<f32>{
    let mut out = Vec::new();

    let std_dev = (2.0 / fan_in as f32).sqrt();
    let normal = Normal::new(0.0_f32, std_dev).unwrap();
    let mut rng = thread_rng();

    for _ in 0..(fan_in * fan_out){
        out.push(normal.sample(&mut rng));
    }

    out
}