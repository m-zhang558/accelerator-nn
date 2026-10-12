"""Measuring how well a model does."""

import torch


def evaluate(model, loader, loss_fn=None):
    """
    Return (accuracy, loss) for the images in `loader`.

    accuracy: the fraction of images the model classifies correctly.
    loss:     the average loss over all the images, computed with `loss_fn`.
              None if no loss_fn is given (e.g. when we only care about accuracy).

    Works for both models, since each one can be called like `model(images)`
    and returns a score (logit) for each of the 10 digits.
    """
    correct = 0
    total = 0
    total_loss = 0.0
    # We are only measuring, not learning, so we don't need gradients.
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)  # the digit with the highest score
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            if loss_fn is not None:
                # loss_fn gives the batch's average, so weight it by the batch size.
                total_loss += loss_fn(logits, labels).item() * labels.size(0)

    accuracy = correct / total
    loss = total_loss / total if loss_fn is not None else None
    return accuracy, loss
