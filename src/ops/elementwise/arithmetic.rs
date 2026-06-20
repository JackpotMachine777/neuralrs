/// Adds two vectors element-wise.
pub fn add_vec(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32>{
    let mut out = vec![0.0; a.len()];

    for i in 0..a.len(){
        out[i] = a[i] + b[i];
    }

    out
}