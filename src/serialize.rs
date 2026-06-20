use crate::nn::linear::Linear;
use crate::tensor::Tensor;
use crate::nn::module::Module;
use std::fs::File;
use std::io::{Write, BufRead, BufReader};

/// Saves a single [`Linear`] layer's weights and bias to a file.
///
/// [`Linear`]: crate::nn::linear::Linear
pub fn save_linear(layer: &Linear, path: &str) {
    let mut file = File::create(path).expect("cannot create file");

    let w = &layer.weights;
    let b = &layer.bias;

    writeln!(file, "weights_shape {} {}", w.shape[0], w.shape[1]).unwrap();
    writeln!(file, "bias_shape {}", b.shape[0]).unwrap();

    writeln!(file, "weights").unwrap();
    for v in &w.storage.data {
        writeln!(file, "{v}").unwrap();
    }

    writeln!(file, "bias").unwrap();
    for v in &b.storage.data {
        writeln!(file, "{v}").unwrap();
    }
}

/// Loads a [`Linear`] layer back from a file written by [`save_linear`].
///
/// [`Linear`]: crate::nn::linear::Linear
pub fn load_linear(path: &str) -> Linear {
    let file = File::open(path).expect("cannot open file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|l| l.unwrap());

    let ws_line = lines.next().unwrap();
    let ws: Vec<usize> = ws_line.split_whitespace().skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    let w_rows = ws[0];
    let w_cols = ws[1];

    let bs_line = lines.next().unwrap();
    let bs: usize = bs_line.split_whitespace().nth(1).unwrap().parse().unwrap();

    let _weights_header = lines.next().unwrap();
    let mut w_data = Vec::with_capacity(w_rows * w_cols);

    for _ in 0..(w_rows * w_cols) {
        w_data.push(lines.next().unwrap().parse::<f32>().unwrap());
    }

    let _bias_header = lines.next().unwrap();
    let mut b_data = Vec::with_capacity(bs);
    
    for _ in 0..bs {
        b_data.push(lines.next().unwrap().parse::<f32>().unwrap());
    }

    Linear {
        weights: Tensor::new(w_data, vec![w_rows, w_cols]),
        bias: Tensor::new(b_data, vec![bs]),
        weights_node: None,
        bias_node: None,
    }
}

/// Saves all of a model's parameters to a file.
///
/// Walks the model's `parameters()` and writes them out, so training can be
/// resumed or the trained weights reused later.
pub fn save_model<M: Module>(model: &mut M, path: &str) {
    let mut file = File::create(path).expect("cannot create file");
    let params = model.parameters();

    writeln!(file, "tensors {}", params.len()).unwrap();

    for p in params {
        writeln!(file, "len {}", p.storage.data.len()).unwrap();

        for v in &p.storage.data {
            writeln!(file, "{v}").unwrap();
        }
    }
}

/// Loads parameters from a file back into a model with matching architecture.
pub fn load_model<M: Module>(model: &mut M, path: &str) {
    let file = File::open(path).expect("cannot open file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|l| l.unwrap());

    let header = lines.next().unwrap();
    let n_tensors: usize = header.split_whitespace().nth(1).unwrap().parse().unwrap();

    let mut params = model.parameters();
    assert_eq!(params.len(), n_tensors, "tensors count in file != model parameters count");

    for p in params.iter_mut() {
        let len_line = lines.next().unwrap();
        let len: usize = len_line.split_whitespace().nth(1).unwrap().parse().unwrap();
        assert_eq!(len, p.storage.data.len(), "tensor size in file != size in model");
        
        for i in 0..len {
            let v: f32 = lines.next().unwrap().parse().unwrap();
            p.storage.data[i] = v;
        }
    }
}