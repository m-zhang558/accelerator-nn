"""
Train the from-scratch MLP (models/mlp_scratch.py) on MNIST.

Run from the repository root:
    python -m train.train_scratch

Everything torch.nn and torch.optim would normally do for us is written out
by hand here: the loss function and the gradient descent update.
PyTorch's autograd (loss.backward()) still computes the gradients.
"""

import argparse
import os

import torch

from models.mlp_scratch import ScratchMLP
from utils.data import get_dataloader
from utils.evaluate import evaluate
from utils.visualize import plot_training

CHECKPOINT_PATH = "weights/mlp_scratch.pt"
PLOT_PATH = "plots/mlp_scratch_training.png"


def cross_entropy(logits, labels):
    """
    Cross-entropy loss: how "surprised" the model is by the correct answers.

    Softmax turns the 10 scores into probabilities:
        p_i = exp(logit_i) / sum_j exp(logit_j)
    and the loss is the average of -log(p_correct) over the batch.
    It is near 0 when the model gives the right digit a probability near 1,
    and large when the right digit gets a low probability.

    We compute log(p_correct) = logit_correct - log(sum_j exp(logit_j)).
    torch.logsumexp does the second part without overflowing on large numbers.
    """
    log_sum_exp = torch.logsumexp(logits, dim=1)                       # [batch]
    correct_logits = logits[torch.arange(logits.shape[0]), labels]     # [batch]
    log_probs_correct = correct_logits - log_sum_exp
    return -log_probs_correct.mean()


def sgd_step(parameters, lr):
    """
    One step of gradient descent: nudge every parameter slightly in the
    direction that lowers the loss, then reset its gradient for the next batch.
    """
    # Updating the weights is not part of the model's computation,
    # so we tell autograd not to track it.
    with torch.no_grad():
        for param in parameters:
            param -= lr * param.grad
            # PyTorch adds new gradients onto old ones, so we clear them.
            param.grad.zero_()


def main():
    parser = argparse.ArgumentParser(description="Train the from-scratch MLP on MNIST.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Fix the random seed so every run gives the same results.
    torch.manual_seed(0)

    train_loader = get_dataloader("train", args.batch_size)
    val_loader = get_dataloader("val", batch_size=1000)
    model = ScratchMLP()

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_val_accuracy = 0.0
    train_losses, val_losses, val_accuracies = [], [], []  # one entry per epoch, for the training plot

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for images, labels in train_loader:
            logits = model(images)                # 1. forward pass: make predictions
            loss = cross_entropy(logits, labels)  # 2. measure how wrong they are
            loss.backward()                       # 3. backward pass: compute gradients
            sgd_step(model.parameters(), args.lr) # 4. update the weights
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy, val_loss = evaluate(model, val_loader, cross_entropy)
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
    plot_training(train_losses, val_losses, val_accuracies, "From-scratch MLP training", PLOT_PATH)


if __name__ == "__main__":
    main()
