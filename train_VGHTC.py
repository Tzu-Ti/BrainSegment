__author__ = 'Titi'
import argparse
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
import os
import nrrd
import json
import numpy as np
import glob

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
from data.dataset import VGHTCDataset, VGHTCDatasetNoSeg

from utils import Unormalize
from utils import MeasureMetric
from utils import DiceLoss

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.0005)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--predict', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--ckpt_path')
    # I/O
    parser.add_argument('--root', default='Hippo_dataset_VGHTC_share')
    parser.add_argument('--trainMask_path', default='Hippo_dataset_VGHTC_share/trainList.txt')
    parser.add_argument('--testMask_path', default='Hippo_dataset_VGHTC_share/testList.txt')
    parser.add_argument('--patient_name')
    parser.add_argument('--save_result', action="store_true")
    # model setting
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--class_num', type=int, default=44)
    parser.add_argument('--down_times', type=int, default=5)
    return parser.parse_args()


class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.UNet = UNet(n_classes=args.class_num)

        # training setting
        self.CE = nn.CrossEntropyLoss()
        self.DICE = DiceLoss()

        self.LMM = MeasureMetric(metrics=['DICE'])
        self.RMM = MeasureMetric(metrics=['DICE'])
        self.MM = MeasureMetric(metrics=['DICE'])

        with open(args.labelmap_path) as f:
            self.LabelMap = json.load(f)

    def searchKey(self, dict, value):
        for k, v in dict.items():
            if v == value:
                return k

    def Mappingback(self, seg):
        new = np.zeros_like(seg, dtype=np.int64)
        exist_label = list(self.LabelMap.values())
        for label in range(1, self.args.class_num):
            if label not in exist_label:
                origin_label = 0
            else:
                origin_label = int(self.searchKey(self.LabelMap, label))
            new += np.where(seg==label, origin_label, 0)
        return new
    
    def show_seg(self, x):
        if not x.dtype == torch.long: # if not ground truth, that is prediction, need to do softmax
            x = F.softmax(x, dim=1)
        n = torch.argmax(x, dim=1, keepdims=True).type(torch.uint8)
        n = n * 5
        return n

    def training_step(self, batch, batch_idx):
        img, seg = batch
        output = self.UNet(img)
        
        CE = self.CE(output, seg.type(torch.float32))
        loss = CE
        ################################################
        # Log
        self.log_dict({
            'Loss': loss
        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_images("Train img", Unormalize(img[:4]), batch_idx)
        seg_vis = self.show_seg(seg)
        self.logger.experiment.add_images("Train GT", seg_vis[:4], batch_idx)
        output_vis = self.show_seg(output)
        self.logger.experiment.add_images("Train output", output_vis[:4], batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        img, seg = batch
        output = self.UNet(img)

        # Metric
        s = F.softmax(output, dim=1)
        s = torch.where(s > 0.5, 1, 0)
        self.MM.update(s, seg)

        # visualization
        self.visual_img = Unormalize(img[:4])
        seg_vis = self.show_seg(seg)
        self.visual_seg = seg_vis[:4]
        output_vis = self.show_seg(output)
        self.visual_output = output_vis[:4]

    def on_validation_epoch_end(self):
        M = self.MM.compute()
        self.log_dict({
            'all': M['DICE']
        }, on_epoch=True, sync_dist=True)
        
        self.logger.experiment.add_images("Val images", self.visual_img, self.current_epoch)
        self.logger.experiment.add_images("Val GT", self.visual_seg, self.current_epoch)
        self.logger.experiment.add_images("Val output", self.visual_output, self.current_epoch)

    def test_step(self, batch, batch_idx):
        img, seg = batch
        output = self.UNet(img)

        # Metric
        s = F.softmax(output, dim=1)
        s = torch.where(s > 0.5, 1, 0)
        self.MM.update(s, seg)

        pred = F.softmax(output, dim=1)
        pred = torch.argmax(pred, dim=1, keepdims=True).type(torch.uint8)

        if self.args.save_result:
            pred = pred.squeeze(1)
            if batch_idx == 0:
                self.prediction = pred
            else:
                self.prediction = torch.cat([self.prediction, pred], dim=0)

    def on_test_end(self):
        M = self.MM.compute()
        results = {
            'all': M['DICE']
        }
        print(results)

        if self.args.save_result:
            pred = self.prediction.permute(1, 2, 0).cpu().detach().numpy()
            pred = self.Mappingback(pred)

            folder = "Output/{}".format(self.args.patient_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            MR, header = nrrd.read(os.path.join(self.args.root, self.args.patient_name, 'MR.nrrd'))
            nrrd.write(os.path.join(folder, 'prediction.seg.nrrd'), pred,
                       header=header)
            
    def predict_step(self, batch, batch_idx):
        img = batch
        output = self.UNet(img)

        pred = F.softmax(output, dim=1)
        pred = torch.argmax(pred, dim=1, keepdims=True).type(torch.uint8)

        if self.args.save_result:
            pred = pred.squeeze(1)
            if batch_idx == 0:
                self.prediction = pred
            else:
                self.prediction = torch.cat([self.prediction, pred], dim=0)
    
    def on_predict_end(self):
        if not self.args.save_result:
            return
        pred = self.prediction.permute(1, 2, 0).cpu().detach().numpy()
        pred = self.Mappingback(pred)

        folder = "Output/{}".format(self.args.patient_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        MR, header = nrrd.read(os.path.join(self.args.root, self.args.patient_name, 'MR.nrrd'))
        nrrd.write(os.path.join(folder, 'prediction.seg.nrrd'), pred,
                    header=header)
    
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

    trainDataset = VGHTCDataset(folder=args.root, lst=args.trainMask_path, size=args.resolution, num_classes=args.class_num, train=True)
    trainDataloader = DataLoader(dataset=trainDataset,
                                 batch_size=args.batch_size,
                                 shuffle=True, 
                                 num_workers=cpu_count())
    valDataset = VGHTCDataset(folder=args.root, lst=args.testMask_path, size=args.resolution, num_classes=args.class_num, train=False)
    valDataloader = DataLoader(dataset=valDataset,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=cpu_count())
    testDataset = VGHTCDataset(folder=args.root, patient_name=args.patient_name, size=args.resolution, num_classes=args.class_num, train=False)
    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=cpu_count())

    Model = Model_factory(args)
    
    # trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, accelerator='gpu', devices=[1])
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=5,
                         logger=tb_logger, log_every_n_steps=5)
    if args.train:
        # trainer.fit(model=Model, train_dataloaders=trainDataloader)
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)
    elif args.predict:
        predictDataset = VGHTCDatasetNoSeg(folder=args.root, patient_name=args.patient_name, size=args.resolution, num_classes=args.class_num)
        predictDataloader = DataLoader(dataset=predictDataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count())
        predict_trainer = pl.Trainer(devices=1, logger=tb_logger)
        predict_trainer.predict(model=Model, dataloaders=predictDataloader, ckpt_path=args.ckpt_path)
              
if __name__ == '__main__':
    main()
