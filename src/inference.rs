use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};

use crate::data::MnistBatcher;
use crate::training::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;

    // From burn_app
//    let batcher = MnistBatcher::new(device);
//    let batch = batcher.batch(vec![item]);

    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    // let output = model.forward(batch.images);
    let output: Tensor<B, 2> = model.forward(batch.images);
    println!("Output tensor before argmax");
    println!("{:#?}", output);
    println!("With ownership {:#?}", output.clone().into_data()); // Have to clone as used 'into'
    println!("Without {:#?}", output.to_data());
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
