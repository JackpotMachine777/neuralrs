use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Xavier/Glorot initialization — random weights scaled for tanh/sigmoid-style
/// activations.
///
/// Scales by `fan_in` and `fan_out` together to keep variance stable through
/// symmetric activations. Returns a flat vector of `fan_in * fan_out` values.
pub fn xavier(fan_in: usize, fan_out: usize) -> Vec<f32>{
    let mut out = Vec::new();

    let std_dev = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let normal = Normal::new(0.0_f32, std_dev).unwrap();
    let mut rng = thread_rng();

    for _ in 0..(fan_in * fan_out){
        out.push(normal.sample(&mut rng));
    }

    out
}