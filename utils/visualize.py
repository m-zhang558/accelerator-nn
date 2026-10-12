"""Plotting predictions on test images, and how training went."""

import os

import matplotlib.pyplot as plt
import torch

from utils.data import unnormalize


def show_predictions(model, dataset, num_images=16, only_mistakes=False, save_path=None):
    """
    Show a grid of test images, each labeled with the model's prediction and the
    true digit. Wrong predictions are shown in red.
    """
    images, labels, predictions = [], [], []
    with torch.no_grad():
        for image, label in dataset:
            prediction = model(image.unsqueeze(0)).argmax(dim=1).item()  # add a batch dimension of 1
            if only_mistakes and prediction == label:
                continue
            images.append(image)
            labels.append(label)
            predictions.append(prediction)
            if len(images) == num_images:
                break

    cols = 4
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, label, prediction in zip(axes.flat, images, labels, predictions):
        ax.imshow(unnormalize(image).squeeze(), cmap="gray")
        color = "green" if prediction == label else "red"
        ax.set_title(f"pred {prediction} / true {label}", color=color, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# Colors for the training plots. Each dataset keeps its color in both panels.
TRAIN_COLOR = "#2a78d6"   # blue
VAL_COLOR = "#eb6834"     # orange
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def plot_training(train_losses, val_losses, val_accuracies, title, save_path):
    """
    Save a picture of how training went, after each epoch:
      left:  training loss and validation loss
      right: validation accuracy, with the epoch whose weights were saved
             (the best validation accuracy) marked
    """
    epochs = list(range(1, len(train_losses) + 1))
    best_index = max(range(len(val_accuracies)), key=lambda i: val_accuracies[i])
    best_epoch = best_index + 1
    val_percent = [accuracy * 100 for accuracy in val_accuracies]

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13)

    loss_ax.plot(epochs, train_losses, color=TRAIN_COLOR, linewidth=2, marker="o", markersize=5, label="Training loss")
    loss_ax.plot(epochs, val_losses, color=VAL_COLOR, linewidth=2, marker="o", markersize=5, label="Validation loss")
    # Label each line at its last point as well as in the legend.
    for losses, name, color in [(train_losses, "train", TRAIN_COLOR), (val_losses, "val", VAL_COLOR)]:
        loss_ax.annotate(name, (epochs[-1], losses[-1]), xytext=(8, 0), textcoords="offset points",
                         va="center", color=TEXT_PRIMARY, fontsize=9)
    loss_ax.set_title("Loss (lower is better)", color=TEXT_PRIMARY, fontsize=11)
    loss_ax.set_ylim(bottom=0)
    loss_ax.set_xlim(0.5, len(epochs) + 1)
    loss_ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    acc_ax.plot(epochs, val_percent, color=VAL_COLOR, linewidth=2, marker="o", markersize=5)
    acc_ax.plot(best_epoch, val_percent[best_index], marker="o", markersize=10,
                markerfacecolor="white", markeredgecolor=VAL_COLOR, markeredgewidth=2)
    acc_ax.annotate(f"best: {val_percent[best_index]:.2f}%\n(epoch {best_epoch}, saved)",
                    (best_epoch, val_percent[best_index]), xytext=(0, -38), textcoords="offset points",
                    ha="center", color=TEXT_PRIMARY, fontsize=9)
    acc_ax.set_title("Validation accuracy % (higher is better)", color=TEXT_PRIMARY, fontsize=11)

    for ax in (loss_ax, acc_ax):
        ax.set_xlabel("Epoch", color=TEXT_SECONDARY)
        ax.set_xticks(epochs)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training plot to {save_path}")
