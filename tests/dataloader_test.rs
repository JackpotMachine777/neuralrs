use neuralrs::data::dataloader::DataLoader;

#[test]
fn dataloader_batching() {
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
        vec![5.0, 5.0],
    ];
    let targets = vec![
        vec![10.0],
        vec![20.0],
        vec![30.0],
        vec![40.0],
        vec![50.0],
    ];

    let loader = DataLoader::new(inputs, targets, 2);

    assert_eq!(loader.len(), 5);
    assert_eq!(loader.num_batches(), 3);

    let (in0, tgt0, sz0) = loader.get_batch(0);
    assert_eq!(sz0, 2);
    assert_eq!(in0, vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(tgt0, vec![10.0, 20.0]);

    let (in2, tgt2, sz2) = loader.get_batch(2);
    assert_eq!(sz2, 1);
    assert_eq!(in2, vec![5.0, 5.0]);
    assert_eq!(tgt2, vec![50.0]);

    println!("batching ok");
}

#[test]
fn dataloader_shuffle_keeps_pairs() {
    let inputs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32]).collect();
    let targets: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 10.0]).collect();

    let mut loader = DataLoader::new(inputs, targets, 1);
    loader.shuffle();

    for b in 0..loader.num_batches() {
        let (inp, tgt, _) = loader.get_batch(b);
        assert_eq!(tgt[0], inp[0] * 10.0, "shuffle paired input and target!");
    }

    println!("shuffle keeps pairs ok");
}