import torch

import numpy as np

# PyTroch version

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.cpu().detach().numpy().squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels.cpu().detach().numpy()).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs, labels):
    pred = torch.sigmoid(outputs)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.cpu().detach().numpy().squeeze(1).astype(int)
    labels = labels.cpu().detach().numpy().astype(int)

    intersection = np.bitwise_and(pred, labels).sum((1, 2))
    union = (pred | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded.mean()  # Or thresholded.mean()