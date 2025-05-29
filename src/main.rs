#![recursion_limit = "256"] // needed for Vulkan
mod model;
mod data;
mod training;
mod inference;

// use burn::prelude::*;

use burn::backend::{Autodiff, Vulkan};
// use burn::backend::Wgpu;
use burn::tensor::Tensor;

use crate::model::ModelConfig;
use std::time::Instant;
use burn::optim::AdamConfig;
use crate::training::TrainingConfig;
use burn::data::dataloader::Dataset;

fn main() {
    println!("Try burn");
    let before = Instant::now();

    // type MyBackend = Wgpu<f32, i32>; // todo change to Vulkan for an attempt
    type MyBackend = Vulkan;
    // let device = Default::default();
    let device = burn::backend::wgpu::WgpuDevice::default();
    // let devicce = burn::backend::Vulkan::VulkanDevice::default();

    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<MyBackend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<MyBackend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);

    println!("\nSetup model"); // todo try Backend here instead of MyBackend and timeit
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    // Print out models configuration
    println!("{}", model);

    println!("Train");
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    println!("Inference time");
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );

    let duration = before.elapsed();
    println!("\nProcessing took: {duration:.2?}");
}
