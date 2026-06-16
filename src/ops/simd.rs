#[cfg(target_arch = "x86_64")]
pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += a[tail_start + i] * b[tail_start + i];
        }

        sum
    }
}