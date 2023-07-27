import torch
from torch import nn
from torch.nn import functional as F

def Unormalize(x):
    x = x.clip(-1, 1)
    x = (x+1) / 2
    return x

import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import dice
from torch import nn

class MeasureMetric(pl.LightningModule):
    def __init__(self, metrics=['FID', 'SSIM']):
        super().__init__()
        self.metrics = metrics
        for m in metrics:
            if m != 'FID': # fid don't need calc mean after all data
                setattr(self, m, MeanMetric())

        if 'FID' in metrics:
            self.FrechetInceptionDistance = FrechetInceptionDistance(feature=2048, normalize=True)
            self.n_fid = 0

    def update(self, fake, real):
        B, C, H, W = fake.shape
        if 'SSIM' in self.metrics:
            ssim = structural_similarity_index_measure(fake, real, data_range=1.0)
            self.SSIM.update(ssim)
        if 'PSNR' in self.metrics:
            psnr = peak_signal_noise_ratio(fake, real, data_range=1.0)
            self.PSNR.update(psnr)
        if 'FID' in self.metrics and self.n_fid < 10_000:
            self.n_fid += B
            self.FrechetInceptionDistance.update(real, real=True)
            self.FrechetInceptionDistance.update(fake, real=False)
        if 'DICE' in self.metrics:
            d = dice(fake, real, average='micro')
            self.DICE.update(d)

    def compute(self):
        results = {}
        for m in self.metrics:
            if m != 'FID':
                meanmetric = getattr(self, m)
                results[m] = meanmetric.compute()
        if 'FID' in self.metrics:
            results['FID'] = self.FrechetInceptionDistance.compute()
            self.n_fid = 0

        return results
    
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
    
    
# smooth = 1e-5
# input = torch.sigmoid(input)
# num = target.size(0)
# input = input.view(num, -1)
# target = target.view(num, -1)
# intersection = (input * target)
# dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
# dice = 1 - dice.sum() / num