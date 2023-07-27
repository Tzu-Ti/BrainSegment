import torch
from torch import nn
from torch.nn import functional as F

def Unormalize(x):
    x = x.clip(-1, 1)
    x = (x+1) / 2
    return x

import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics import Metric

class MeasureMetric(pl.LightningModule):
    def __init__(self, metrics=['DICE']):
        super().__init__()
        self.metrics = metrics
        for m in metrics:
            setattr(self, m, MeanMetric())

    def update(self, pred, target):
        if 'DICE' in self.metrics:
            d = dice(pred, target)
            self.DICE.update(d)

    def compute(self):
        results = {}
        for m in self.metrics:
            meanmetric = getattr(self, m)
            results[m] = meanmetric.compute()

        return results

def dice(pred, target):
    smooth = 1e-5
    B = pred.shape[0]
    m1 = pred.view(B, -1)
    m2 = target.view(B, -1)
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5
        
    def forward(self, input, target):
        B, C, H, W = target.shape
        input = F.softmax(input, dim=1)
        input = torch.flatten(input, 1, -1)
        print(input.shape)
        target = torch.flatten(target, 1, -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + smooth) / (input.sum(1) + smooth)
        dice = 1 - dice.sum() / B
        raise
        return dice