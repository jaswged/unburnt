use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

// This batcher is pretty straightforward, as it only defines a struct that will implement the Batcher trait.
#[derive(Clone, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

// Implement the batching logic
impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            // for each item, convert the image to float32 data struct
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            // for each data struct, create a tensor on the device
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            // for each tensor, reshape to the image dimensions [C, H, W]
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // for each image tensor, apply normalization: make between [0,1] and make the mean =  0 and std = 1
            // Normalize: scale between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            // consume the resulting iterator & collect the values into a new vector
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
