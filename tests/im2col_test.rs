use rstorch::ops::im2col::im2col;

#[test]
fn im2col_basic_test() {
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];

    let (col, col_h, col_w) = im2col(&data, 1, 1, 3, 3, 2, 2, 1, 0);

    println!("col_h={}, col_w={}", col_h, col_w);
    println!("col: {:?}", col);

    assert_eq!(col_h, 4);
    assert_eq!(col_w, 4);

    let expected = vec![
        1.0, 2.0, 4.0, 5.0,
        2.0, 3.0, 5.0, 6.0,
        4.0, 5.0, 7.0, 8.0,
        5.0, 6.0, 8.0, 9.0,
    ];

    assert_eq!(col, expected);
    println!("im2col ok");
}