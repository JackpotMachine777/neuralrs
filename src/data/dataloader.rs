use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataLoader {
    pub inputs: Vec<Vec<f32>>,
    pub targets: Vec<Vec<f32>>,
    pub batch_size: usize,
    pub order: Vec<usize>,
    pub augment: Option<Box<dyn Fn(&[f32]) -> Vec<f32>>>,
    pub training: bool,
}

impl DataLoader {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, batch_size: usize) -> Self {
        assert_eq!(inputs.len(), targets.len(), "input and targets must contain the same amount of examples");
        let n = inputs.len();

        DataLoader {
            inputs,
            targets,
            batch_size,
            order: (0..n).collect(),
            augment: None,
            training: true,
        }
    }

    pub fn len(&self) -> usize { self.inputs.len() }

    pub fn num_batches(&self) -> usize { (self.inputs.len() + self.batch_size - 1) / self.batch_size }

    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.order.shuffle(&mut rng);
    }

    pub fn set_augment(&mut self, f: Box<dyn Fn(&[f32]) -> Vec<f32>>) {
        self.augment = Some(f);
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn get_batch(&self, b: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let start = b * self.batch_size;
        let end = ((b + 1) * self.batch_size).min(self.inputs.len());
        let actual_size = end - start;

        let mut batch_in = Vec::new();
        let mut batch_tgt = Vec::new();

        for i in start..end {
            let idx = self.order[i];

            if self.training {
                if let Some(aug) = &self.augment {
                    let augmented = aug(&self.inputs[idx]);
                    batch_in.extend_from_slice(&augmented);
                } else {
                    batch_in.extend_from_slice(&self.inputs[idx]);
                }
            } else {
                batch_in.extend_from_slice(&self.inputs[idx]);
            }

            batch_tgt.extend_from_slice(&self.targets[idx]);
        }

        (batch_in, batch_tgt, actual_size)
    }
}