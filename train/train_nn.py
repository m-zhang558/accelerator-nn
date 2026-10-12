"""
Train the torch.nn MLP (models/mlp_nn.py) on MNIST.

Run from the repository root:
    python -m train.train_nn

This is the same training loop as train/train_scratch.py, but the loss
function and the weight update come from PyTorch instead of being hand-written.
"""

import argparse
import os

import torch
from torch import nn

from models.mlp_nn import MLP
from utils.data import get_dataloader
from utils.evaluate import evaluate
from utils.visualize import plot_training

CHECKPOINT_PATH = "weights/mlp_nn.pt"
PLOT_PATH = "plots/mlp_nn_training.png"


def main():
    parser = argparse.ArgumentParser(description="Train the torch.nn MLP on MNIST.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Fix the random seed so every run gives the same results.
    torch.manual_seed(0)

    train_loader = get_dataloader("train", args.batch_size)
    val_loader = get_dataloader("val", batch_size=1000)
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()                                    # replaces cross_entropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)        # replaces sgd_step()

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_val_accuracy = 0.0
    train_losses, val_losses, val_accuracies = [], [], []  # one entry per epoch, for the training plot

    for epoch in range(1, args.epochs + 1):
        model.train()  # put the model in training mode (matters for layers like dropout)
        total_loss = 0.0
        for images, labels in train_loader:
            logits = model(images)            # 1. forward pass: make predictions
            loss = loss_fn(logits, labels)    # 2. measure how wrong they are
            optimizer.zero_grad()             #    clear gradients from the last batch
            loss.backward()                   # 3. backward pass: compute gradients
            optimizer.step()                  # 4. update the weights
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        model.eval()  # put the model in evaluation mode before measuring accuracy
        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch:2d} | train loss {avg_loss:.4f} | val loss {val_loss:.4f} | val accuracy {val_accuracy:.2%}", end="")

        # Accuracy jumps around from epoch to epoch, so the last epoch isn't always the best.
        # Keep the weights from whichever epoch did best on the validation set.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  <- best so far, saved", end="")
        print()

    print(f"Best val accuracy {best_val_accuracy:.2%}, weights saved to {CHECKPOINT_PATH}")
    plot_training(train_losses, val_losses, val_accuracies, "torch.nn MLP training", PLOT_PATH)


if __name__ == "__main__":
    main()
