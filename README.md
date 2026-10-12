# Neural Networks: Classifying Handwritten Digits

A beginner lesson on neural networks. We train a small network to recognize handwritten
digits (0-9) from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

The same network is built twice:

1. **From scratch**: raw PyTorch tensors only, no `torch.nn`. Every weight, the loss
   function and the gradient descent update are written out by hand.
2. **With `torch.nn`**: the same network using PyTorch's building blocks, the way
   you'd write it in practice.

Both reach about **98% test accuracy** in under a minute on a laptop CPU.

## The network

```
image (28x28 pixels) -> flatten -> [784]
    -> Linear -> [128] -> ReLU
    -> Linear -> [64]  -> ReLU
    -> Linear -> [10]     one score per digit; the highest score is the prediction
```

## Project layout

```
models/       what the network is
  mlp_scratch.py    the MLP built from raw tensors
  mlp_nn.py         the same MLP built with torch.nn
train/        how the network learns
  train_scratch.py  hand-written loss and gradient descent
  train_nn.py       nn.CrossEntropyLoss and torch.optim.SGD
inference/    using a trained network
  predict_scratch.py
  predict_nn.py
utils/        code shared by everything above
  data.py           downloads MNIST and splits it into train / val / test
  evaluate.py       measures accuracy
  visualize.py      plots predictions
datasets/     MNIST is downloaded here (created automatically)
checkpoints/  trained weights are saved here (created automatically)
```

## The data

MNIST's 70,000 images are downloaded once into `datasets/` (about 66 MB). They are stored
as a few binary files rather than separate pictures: each image is 784 bytes (28 x 28
pixels, one byte per pixel from 0 = black to 255 = white), packed one after another.

`utils/data.py` splits them three ways in code:

| Split | Images | Used for |
|---|---|---|
| `train` | 50,000 | the only images the model learns from |
| `val` | 10,000 | checked after every epoch while training |
| `test` | 10,000 | the final score, used only by `inference/` |

- **Validation**: checked after every epoch. The training scripts save the weights from the
  epoch with the best validation accuracy, since accuracy jumps around and the last epoch
  isn't always the best. You also use it to decide things like the learning rate. Because
  you pick whatever looks good on it, it slowly stops being a fair test.
- **Test**: kept aside until the end, so the final accuracy is an honest measure of how the
  model does on images it has never seen.

The validation set is a random 10,000 of MNIST's official 60,000 training images, chosen
with a fixed seed so it's the same every run. The test set is MNIST's official 10,000 test
images.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Run every command **from the repository root**, using `python -m` so the imports between
folders work.

```bash
# Train (downloads MNIST the first time; prints validation accuracy after every epoch)
python -m train.train_scratch
python -m train.train_nn

# Measure test accuracy and see predictions from the trained models
python -m inference.predict_scratch
python -m inference.predict_nn --mistakes    # only show the digits it got wrong
```

Training scripts accept `--epochs`, `--lr` (learning rate) and `--batch-size`, for example:

```bash
python -m train.train_nn --epochs 5 --lr 0.01
```

## Suggested reading order

1. `utils/data.py`: what the data looks like and how it's split.
2. `models/mlp_scratch.py`, then `train/train_scratch.py`: every step done by hand.
3. `models/mlp_nn.py`, then `train/train_nn.py`: the same thing with library pieces.
   Compare them side by side.
4. `inference/`: loading a trained model and using it.

## Things to try

- Change the hidden layer sizes (128 and 64). Does a bigger network do better?
- Try learning rates of `1.0`, `0.01` and `0.001`. What happens to the loss?
- Train for 1 epoch vs. 20 epochs. When does validation accuracy stop improving?
- Look at the `--mistakes` output. Would you have gotten those digits right?
