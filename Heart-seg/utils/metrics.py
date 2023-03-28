import torch
import torch.nn as nn
import numpy as np


def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N

def accuracy(pred, gt):
    """(TP + TN) / (TP + FP + FN + TN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp + tn).float() / (tp + fp + tn + fn).float()

    return score.sum() / N


def precision(pred, gt):
    """TP / (TP + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))

    score = tp.float() / (tp + fp).float()

    return score.sum() / N


def sensitivity(pred, gt):
    """TP / (TP + FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = tp.float() / (tp +  fn).float()

    return score.sum() / N


def specificity(pred, gt):
    """TN / (TN + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))

    score = tn.float() / (fp + tn).float()

    return score.sum() / N


def recall(pred, gt):

    return sensitivity(pred, gt)
