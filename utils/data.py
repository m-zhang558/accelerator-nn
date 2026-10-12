"""
Loading the MNIST dataset.

MNIST is 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels.
It officially comes as 60,000 training images and 10,000 test images. We split it
three ways:

    train   50,000 images: the model learns from these
    val     10,000 images: checked after each epoch while training (taken from the official training images)
    test    10,000 images: the final measure of accuracy (the official test images)

Why three sets?
  - We tune choices (epochs, learning rate, layer sizes) by watching validation accuracy.
  - Because those choices were picked to look good on the validation set, it is no longer
    a fair test. The test set is kept aside so we get an honest final score.

Both models use this same file, so they see exactly the same data.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATASETS_DIR = "datasets"
VAL_SIZE = 10_000

# The average and standard deviation of all MNIST pixel values (after scaling to 0-1).
# Subtracting the mean and dividing by the std centers the inputs around 0,
# which makes training faster and more stable.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                         # image (0-255) -> tensor of shape [1, 28, 28] with values 0-1
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def get_dataset(split):
    """
    Return the "train", "val" or "test" set. MNIST is downloaded the first time.

    dataset[i] returns one (image, label) pair.
    """
    if split == "test":
        return datasets.MNIST(DATASETS_DIR, train=False, download=True, transform=TRANSFORM)

    # Carve the validation set out of the official training images.
    # random_split shuffles before splitting, so both sets get a fair mix of digits
    # and handwriting styles. The fixed seed makes the split the same every time,
    # so no image ever moves between train and val.
    official_train = datasets.MNIST(DATASETS_DIR, train=True, download=True, transform=TRANSFORM)
    train_set, val_set = random_split(
        official_train,
        [len(official_train) - VAL_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(0),
    )
    return {"train": train_set, "val": val_set}[split]


def get_dataloader(split, batch_size=64):
    """
    Return a DataLoader, which hands us the data in small batches.

    Each batch is a pair (images, labels):
      images: tensor of shape [batch_size, 1, 28, 28]
      labels: tensor of shape [batch_size], each a digit 0-9
    """
    # Shuffle the training data so every epoch sees the images in a new order.
    # The order doesn't matter when we're only measuring accuracy.
    return DataLoader(get_dataset(split), batch_size=batch_size, shuffle=(split == "train"))


def unnormalize(image):
    """Undo the normalization so an image can be displayed with normal brightness."""
    return image * MNIST_STD + MNIST_MEAN
