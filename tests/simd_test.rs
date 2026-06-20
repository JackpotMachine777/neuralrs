use neuralrs::ops::simd::dot_simd;

#[test]
fn dot_simd_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let result = dot_simd(&a, &b);
    println!("dot (len 8): {result}");
    assert!((result - 36.0).abs() < 1e-4);
}

#[test]
fn dot_simd_with_tail() {
    let a: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let b = vec![2.0; 10];
    let result = dot_simd(&a, &b);
    println!("dot (len 10): {result}");
    assert!((result - 110.0).abs() < 1e-4);
}

#[test]
fn dot_simd_matches_naive() {
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.3).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.2).collect();

    let simd = dot_simd(&a, &b);
    let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    println!("simd: {simd}, naive: {naive}");
    assert!((simd - naive).abs() < 1e-2); 
}