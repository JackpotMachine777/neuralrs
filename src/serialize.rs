use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::tensor::Tensor;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

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

/// Saves raw tensors to a text checkpoint at `path`.
///
/// The format matches [`save_model`]: a `tensors N` header, then for each
/// tensor a `len L` line followed by one value per line. Values are written
/// with `Display`, which prints the shortest representation that round-trips
/// exactly, so the checkpoint is lossless.
///
/// The data is first written to `<path>.tmp` and atomically renamed into
/// place, so an interrupted save (Ctrl-C, crash) can never leave a truncated
/// checkpoint behind, the previous one stays intact.
pub fn save_tensors(tensors: &[&[f32]], path: &str) {
    let tmp = format!("{path}.tmp");
    {
        let file = File::create(&tmp).expect("save_tensors: cannot create file");
        let mut w = BufWriter::new(file);

        writeln!(w, "tensors {}", tensors.len()).unwrap();
        for t in tensors {
            writeln!(w, "len {}", t.len()).unwrap();
            for v in *t {
                writeln!(w, "{v}").unwrap();
            }
        }
        w.flush().expect("save_tensors: flush failed");
    }
    fs::rename(&tmp, path).expect("save_tensors: rename failed");
}

/// Loads tensors from a checkpoint written by [`save_tensors`] (or by
/// [`save_model`], the formats are identical).
///
/// Returns the tensors in file order; the caller is responsible for checking
/// that counts and lengths match its model.
pub fn load_tensors(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("load_tensors: cannot open file");
    let mut lines = BufReader::new(file)
        .lines()
        .map(|l| l.expect("load_tensors: read failed"));

    let header = lines.next().expect("load_tensors: missing header");
    let n: usize = header
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .expect("load_tensors: malformed header");

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len_line = lines
            .next()
            .expect("load_tensors: file ends before all tensors were read");
        let len: usize = len_line
            .split_whitespace()
            .nth(1)
            .and_then(|x| x.parse().ok())
            .expect("load_tensors: malformed tensor header");

        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let line = lines.next().expect("load_tensors: file ends mid-tensor");
            data.push(line.parse().expect("load_tensors: malformed value"));
        }
        out.push(data);
    }
    out
}

/// Saves all of a model's parameters to a file.
///
/// Walks the model's `parameters()` and writes them out through
/// [`save_tensors`], so training can be resumed or the trained weights reused
/// later. The write is buffered and atomic, see [`save_tensors`].
pub fn save_model<M: Module>(model: &mut M, path: &str) {
    let params = model.parameters();
    let refs: Vec<&[f32]> = params.iter().map(|p| p.storage.data.as_slice()).collect();
    save_tensors(&refs, path);
}

/// Loads parameters from a file back into a model with matching architecture.
pub fn load_model<M: Module>(model: &mut M, path: &str) {
    let loaded = load_tensors(path);
    let mut params = model.parameters();
    assert_eq!(params.len(), loaded.len(), "tensors count in file != model parameters count");

    for (p, data) in params.iter_mut().zip(loaded) {
        assert_eq!(data.len(), p.storage.data.len(), "tensor size in file != size in model");
        p.storage.data.copy_from_slice(&data);
    }
}

/// A named tensor for [`save`]: `(name, shape, data)`.
pub type NamedTensor<'a> = (&'a str, &'a [usize], &'a [f32]);

/// Saves named tensors to `path`, picking the format from the extension:
/// `.safetensors` writes the binary safetensors format (names, shapes, and
/// dtype preserved, loadable from PyTorch and friends); anything else writes
/// the positional text format of [`save_tensors`], dropping names and shapes.
/// Both writes are atomic (tmp file + rename), so an interrupted save never
/// truncates an existing checkpoint.
pub fn save(tensors: &[NamedTensor], path: &str) {
    if path.ends_with(".safetensors") {
        save_safetensors(tensors, path);
    } else {
        let raw: Vec<&[f32]> = tensors.iter().map(|(_, _, d)| *d).collect();
        save_tensors(&raw, path);
    }
}

/// Loads tensors from `path`, picking the format from the extension. For
/// `.safetensors` the real names and shapes come back, sorted by name; for
/// the text format names are synthesized as `tensor_{i}` and the shape is
/// the flat length.
pub fn load(path: &str) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    if path.ends_with(".safetensors") {
        load_safetensors(path)
    } else {
        load_tensors(path)
            .into_iter()
            .enumerate()
            .map(|(i, d)| (format!("tensor_{i}"), vec![d.len()], d))
            .collect()
    }
}

fn save_safetensors(tensors: &[NamedTensor], path: &str) {
    let bytes_per: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, d)| d.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let views: Vec<(&str, TensorView)> = tensors
        .iter()
        .zip(&bytes_per)
        .map(|((name, shape, _), bytes)| {
            let view = TensorView::new(Dtype::F32, shape.to_vec(), bytes)
                .expect("save: tensor shape does not match its data length");
            (*name, view)
        })
        .collect();
    let buffer =
        safetensors::serialize(views, None).expect("save: safetensors serialization failed");

    let tmp = format!("{path}.tmp");
    fs::write(&tmp, &buffer).expect("save: cannot write file");
    fs::rename(&tmp, path).expect("save: rename failed");
}

fn load_safetensors(path: &str) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let buffer = fs::read(path).expect("load: cannot read file");
    let st = SafeTensors::deserialize(&buffer).expect("load: not a valid safetensors file");
    let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(
                view.dtype(),
                Dtype::F32,
                "load: only F32 tensors are supported (tensor `{name}`)"
            );
            let data = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (name, view.shape().to_vec(), data)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}