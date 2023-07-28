__author__ = 'Titi'
import argparse
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
import os
import nrrd

from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn import functional as F
import torch.optim as optim

from models.Unet import UNet
from data.dataset import HippoDataset, VGHTCDataset

from utils import Unormalize
from utils import MeasureMetric
from utils import DiceLoss

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=int, default=0.0005)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--ckpt_path')
    # I/O
    parser.add_argument('--root', default='Hippo_25_openDataset')
    parser.add_argument('--trainMask_path', default='data/opendataset/trainMask.txt')
    parser.add_argument('--testMask_path', default='data/opendataset/testMask.txt')
    parser.add_argument('--patient_name')
    parser.add_argument('--save_result', action="store_true")
    # model setting
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--down_times', type=int, default=5)
    return parser.parse_args()


class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.UNet = UNet(n_classes=3)

        # training setting
        self.CE = nn.CrossEntropyLoss()
        self.DICE = DiceLoss()

        self.LMM = MeasureMetric(metrics=['DICE'])
        self.RMM = MeasureMetric(metrics=['DICE'])
        self.MM = MeasureMetric(metrics=['DICE'])
        
    def show_seg(self, output):
        normalized_mask = F.softmax(output, dim=1)
        n = torch.argmax(normalized_mask, dim=1, keepdims=True)
        return n

    def training_step(self, batch, batch_idx):
        img, mask = batch
        gt = mask[:, 0, :, :] + mask[:, 1, :, :] * 2
        output = self.UNet(img)

        mask = mask.squeeze(1)
        CE = self.CE(output, gt)
        loss = CE
        ################################################
        # Log
        self.log_dict({
            'Loss': loss
        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_images("Train img", Unormalize(img[:4]), batch_idx)
        GT = mask[:, 0:1, :, :] + mask[:, 1:2, :, :]
        self.logger.experiment.add_images("Train GT", GT[:4], batch_idx)
        output_vis = self.show_seg(output)
        self.logger.experiment.add_images("Train output", self.show_seg(output)[:4], batch_idx)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        output = self.UNet(img)

        # Metric
        s = F.softmax(output, dim=1)
        s = torch.where(s > 0.5, 1, 0)
        output_L = s[:, 1, :, :]
        output_R = s[:, 2, :, :]
        output_all = output_L + output_R
        self.LMM.update(output_L, mask[:, 0, :, :])
        self.RMM.update(output_R, mask[:, 1, :, :])
        self.MM.update(output_all, mask[:, 0, :, :] + mask[:, 1, :, :])

        # visualization
        self.visual_img = Unormalize(img[:4])
        GT = mask[:, 0:1, :, :] + mask[:, 1:2, :, :]
        self.visual_GT = GT[:4]
        output_vis = self.show_seg(output)
        self.visual_output = output_vis[:4]

    def on_validation_epoch_end(self):
        L = self.LMM.compute()
        R = self.RMM.compute()
        M = self.MM.compute()
        self.log_dict({
            'L': L['DICE'],
            'R': R['DICE'],
            'all': M['DICE']
        }, on_epoch=True, sync_dist=True)
        
        self.logger.experiment.add_images("Val images", self.visual_img, self.current_epoch)
        self.logger.experiment.add_images("Val GT", self.visual_GT, self.current_epoch)
        self.logger.experiment.add_images("Val output", self.visual_output, self.current_epoch)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        output = self.UNet(img)

        # Metric
        s = F.softmax(output, dim=1)
        s = torch.where(s > 0.5, 1, 0)
        output_L = s[:, 1, :, :]
        output_R = s[:, 2, :, :]
        output_all = output_L + output_R
        self.LMM.update(output_L, mask[:, 0, :, :])
        self.RMM.update(output_R, mask[:, 1, :, :])
        self.MM.update(output_all, mask[:, 0, :, :] + mask[:, 1, :, :])

        # rotate back to origin direction
        # output_all = torch.rot90(output_all, k=1, dims=(1, 2))

        if self.args.save_result:
            if batch_idx == 0:
                self.output = output_all
            else:
                self.output = torch.cat([self.output, output_all], dim=0)

    def on_test_end(self):
        L = self.LMM.compute()
        R = self.RMM.compute()
        M = self.MM.compute()
        results = {
            'L': L['DICE'],
            'R': R['DICE'],
            'all': M['DICE']
        }
        print(results)

        if self.args.save_result:
            array = self.output.permute(1, 2, 0).cpu().detach().numpy()
            folder = "Output/{}".format(self.args.patient_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            nrrd.write(os.path.join(folder, 'prediction.nrrd'), array)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.UNet.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]

def main():
    args = parse()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=args.model_name,
        name='lightning_logs'
    )

    trainDataset = HippoDataset(folder=args.root, MaskList_path=args.trainMask_path, train=True)
    trainDataloader = DataLoader(dataset=trainDataset,
                                 batch_size=args.batch_size,
                                 shuffle=True, 
                                 num_workers=cpu_count())
    valDataset = HippoDataset(folder=args.root, MaskList_path=args.testMask_path, train=False)
    valDataloader = DataLoader(dataset=valDataset,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=cpu_count())
    # testDataset = HippoDataset(folder=args.root, MaskList_path=args.testMask_path, train=False)
    testDataset = VGHTCDataset(folder=args.root, patient_name=args.patient_name)
    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=cpu_count())

    Model = Model_factory(args)
    
#     trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, accelerator='gpu', devices=[1, 2])
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=5,
                         logger=tb_logger, log_every_n_steps=5)
    if args.train:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)
              
if __name__ == '__main__':
    main()
