"""
Use the trained from-scratch MLP to classify MNIST test images.

Run from the repository root (after python -m train.train_scratch):
    python -m inference.predict_scratch
    python -m inference.predict_scratch --mistakes   # show only wrong predictions
"""

import argparse

import torch

from models.mlp_scratch import ScratchMLP
from utils.data import get_dataloader, get_dataset
from utils.evaluate import evaluate
from utils.visualize import show_predictions

CHECKPOINT_PATH = "weights/mlp_scratch.pt"


def main():
    parser = argparse.ArgumentParser(description="Run the trained from-scratch MLP on MNIST test images.")
    parser.add_argument("--mistakes", action="store_true", help="only show images the model got wrong")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--save", metavar="PATH", help="save the figure to a file instead of opening a window")
    args = parser.parse_args()

    # Create a model with random weights, then replace them with the trained ones.
    model = ScratchMLP()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    accuracy, _ = evaluate(model, get_dataloader("test", batch_size=1000))
    print(f"Test accuracy: {accuracy:.2%}")

    show_predictions(model, get_dataset("test"), args.num_images, args.mistakes, args.save)


if __name__ == "__main__":
    main()
