use crate::nn::module::Module;

pub fn clip_grad_norm<M: Module>(model: &mut M, max_norm: f32) {
    let params = model.parameters();
    let mut sum_sq = 0.0;

    for p in &params {
        for &g in &p.grad {
            sum_sq += g * g;
        }
    }

    let total_norm = sum_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / total_norm;

        for p in params {
            for g in p.grad.iter_mut() {
                *g *= scale;
            }
        }
    }
}