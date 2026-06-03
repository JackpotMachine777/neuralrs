use rand::thread_rng;
use rand_distr::{Distribution, Normal};

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