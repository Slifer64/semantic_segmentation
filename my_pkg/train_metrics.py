import torch
import numpy as np
from typing import Callable


__all__ = [
    'accuracy',
    'mean_accuracy',
    'jaccard_index',
    'dice_coef',
]

def _reduction_fun(reduction: str) -> Callable:
    if reduction.lower() == 'mean':
        reduce_fun = torch.mean
    elif reduction.lower() == 'none':
        reduce_fun = lambda x: x
    else:
        raise AttributeError('Invalid redction type "%s"' % reduction)
        
    return reduce_fun


def _add_batch_dim(y: torch.Tensor) -> torch.Tensor:
    return y[None, ...] if y.dim() == 3 else y


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    acc = torch.sum(y_true == y_pred) / (y_true.shape[-2] * y_true.shape[-1])
    return acc


def mean_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, reduction='mean', smooth=1e-6):

    y_true = _add_batch_dim(y_true)
    y_pred = _add_batch_dim(y_pred)

    correct_pred_pixels_per_class = torch.sum(torch.abs(y_true * y_pred), dim=[1, 2])
    total_pixels_per_class = torch.sum(y_true, [1, 2])
    acc_per_class = torch.mean(correct_pred_pixels_per_class / total_pixels_per_class, dim=0)
    return _reduction_fun(reduction)(acc_per_class)


def jaccard_index(y_true: torch.Tensor, y_pred: torch.Tensor, reduction='mean', smooth=1e-6) -> torch.Tensor:
    """
    Calculates the Jaccard index (IoU) of the given masks, which are assumed to be in one-hot encoding.
    Each mask must be of shape {B x H x W x K} or {H x W x K}, where B is the batch size and
    K the number of classes.

    Arguments:
        y_true -- torch.Tensor(B, H, W, K), Ground-truth mask.
        y_pred -- torch.Tensor(B, H, W, K), Predicted mask.
        reduction -- "mean": to calc the average IoU over all classes, or
                     "none": to calc the IoU for each class separately
        smooth -- small value to avoid divisions by 0 (optional, default=1e-6)

    Returns:
        torch.Tensor, with either the average IoU over all classes, or, the IoU of each class separately
    """

    y_true = _add_batch_dim(y_true)
    y_pred = _add_batch_dim(y_pred)

    # intersection = torch.sum((y_true & y_pred).float(), dim=[1, 2])
    # union = torch.sum((y_true | y_pred).float(), dim=[1, 2])

    intersection = torch.sum(torch.abs(y_true * y_pred), dim=[1, 2])
    union = torch.sum(y_true, [1, 2]) + torch.sum(y_pred, [1, 2]) - intersection

    iou_per_class = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return _reduction_fun(reduction)(iou_per_class)


def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, reduction='mean', smooth=1e-6) -> torch.Tensor:

    y_true = _add_batch_dim(y_true)
    y_pred = _add_batch_dim(y_pred)

    intersection = torch.sum(y_true * y_pred, dim=[1, 2])
    union = torch.sum(y_true, dim=[1, 2]) + torch.sum(y_pred, dim=[1, 2])
    dice_per_class = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return _reduction_fun(reduction)(dice_per_class)
