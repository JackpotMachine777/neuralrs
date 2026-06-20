/// Interface for learning-rate schedules: given the current step (or epoch),
/// return the learning rate to use.
pub trait Scheduler {
    fn get_lr(&self, step: usize) -> f32;
}

/// Drops the learning rate by a fixed factor every `step_size` steps.
pub struct StepLR {
    pub base_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
}

impl Scheduler for StepLR {
    fn get_lr(&self, step: usize) -> f32 {
        let drops = (step / self.step_size) as i32;
        self.base_lr * self.gamma.powi(drops)
    }
}

/// Multiplies the learning rate by a fixed `gamma` every step — smooth
/// exponential decay.
pub struct ExponentialLR {
    pub base_lr: f32,
    pub gamma: f32,
}

impl Scheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing: the learning rate follows a cosine curve from its starting
/// value down to a minimum over `t_max` steps.
pub struct CosineAnnealingLR {
    pub base_lr: f32,
    pub min_lr: f32,
    pub t_max: usize,
}

impl Scheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        let t = (step.min(self.t_max)) as f32;
        let tmax = self.t_max as f32;
        let cos = (std::f32::consts::PI * t / tmax).cos();
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos)
    }
}

/// Linear warmup followed by cosine annealing.
///
/// Ramps the learning rate up for the first `warmup_steps` (helps stabilize
/// early training), then decays it along a cosine curve down to `min_lr`. This
/// is the schedule used in the MNIST scheduling example.
pub struct WarmupCosine {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub t_max: usize,
}

impl Scheduler for WarmupCosine {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32 + 1.0) / (self.warmup_steps as f32)
        } else {
            let t = (step - self.warmup_steps) as f32;
            let tmax = (self.t_max - self.warmup_steps).max(1) as f32;
            let t = t.min(tmax);
            let cos = (std::f32::consts::PI * t / tmax).cos();
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos)
        }
    }
}