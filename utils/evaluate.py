"""Measuring how accurate a model is."""

import torch


def evaluate(model, loader):
    """
    Return the fraction of images in `loader` that the model classifies correctly.

    Works for both models, since each one can be called like `model(images)`
    and returns a score (logit) for each of the 10 digits.
    """
    correct = 0
    total = 0
    # We are only measuring, not learning, so we don't need gradients.
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)  # the digit with the highest score
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total
