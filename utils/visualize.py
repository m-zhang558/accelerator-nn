"""Plotting a model's predictions on test images."""

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
