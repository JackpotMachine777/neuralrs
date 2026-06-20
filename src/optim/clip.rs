use crate::nn::module::Module;

/// Clips gradients so their combined norm doesn't exceed `max_norm`.
///
/// If the total gradient norm across all the model's parameters is too big,
/// everything gets scaled down proportionally. A common guard against exploding
/// gradients, especially in recurrent models.
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