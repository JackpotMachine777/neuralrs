use crate::tensor::Tensor; 
use crate::ops::matmul::matmul;

/// Unrolls image patches into columns so a convolution can be done as one big
/// matrix multiply (the im2col trick).
pub fn im2col(
    data: &Vec<f32>,
    n: usize,
    c: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let ph = in_h + 2 * pad;
    let pw = in_w + 2 * pad;
    let out_h = (ph - kh) / stride + 1;
    let out_w = (pw - kw) / stride + 1;

    let col_h = c * kh * kw;
    let col_w = out_h * out_w;

    let mut col = vec![0.0; n * col_h * col_w];

    for ni in 0..n {
        for ic in 0..c {
            for i in 0..kh {
                for j in 0..kw {
                    let row = (ic * kh + i) * kw + j;

                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let iy = oy * stride + i;
                            let ix = ox * stride + j;

                            let val = if iy < pad || ix < pad || iy >= pad + in_h || ix >= pad + in_w { 0.0 }
                            else {
                                let oy_in = iy - pad;
                                let ox_in = ix - pad;
                                let src = ((ni * c + ic) * in_h + oy_in) * in_w + ox_in;
                                data[src]
                            };

                            let col_idx = (ni * col_h + row) * col_w + (oy * out_w + ox);
                            col[col_idx] = val;
                        }
                    }
                }
            }
        }
    }

    (col, col_h, col_w)
}

/// Convolution implemented via [`im2col`] plus a matrix multiply.
pub fn conv2d_im2col(
    data: &Vec<f32>,
    weight: &Vec<f32>,
    bias: &Vec<f32>,
    n: usize,
    c_in: usize,
    c_out: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let (col, col_h, col_w) = im2col(data, n, c_in, in_h, in_w, kh, kw, stride, pad);
    let out_h = (in_h + 2 * pad - kh) / stride + 1;
    let out_w = (in_w + 2 * pad - kw) / stride + 1;

    let w_tensor = Tensor::new(weight.clone(), vec![c_out, col_h]);

    let mut out = vec![0.0; n * c_out * col_w];

    for ni in 0..n {
        let start = ni * col_h * col_w;
        let col_slice = col[start..start + col_h * col_w].to_vec();
        let col_tensor = Tensor::new(col_slice, vec![col_h, col_w]);

        let result = matmul(&w_tensor, &col_tensor);
        let res_data = &result.storage.data;

        for oc in 0..c_out {
            for p in 0..col_w {
                let out_idx = (ni * c_out + oc) * col_w + p;
                out[out_idx] = res_data[oc * col_w + p] + bias[oc];
            }
        }
    }

    (out, out_h, out_w)
}