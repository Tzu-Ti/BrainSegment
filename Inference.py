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

from train_VGHTC import Model_factory

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--batch_size', type=int, default=4)
    # I/O
    parser.add_argument('--root', default='/root/VGHTC/hippo/Hippo_dataset_VGHTC_share')
    parser.add_argument('--labelmap_path', default='LabelMap.json')
    parser.add_argument('--patient_name')
    parser.add_argument('--save_result', action="store_true")
    parser.add_argument('--ckpt_path')
    # model setting
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--class_num', type=int, default=44)
    parser.add_argument('--down_times', type=int, default=5)
    return parser.parse_args()

def main():
    args = parse()

    Model = Model_factory.load_from_checkpoint(args.ckpt_path, args=args)

    predictDataset = VGHTCDatasetNoSeg(folder=args.root, patient_name=args.patient_name, size=args.resolution, num_classes=args.class_num)
    predictDataloader = DataLoader(dataset=predictDataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count())
    predict_trainer = pl.Trainer(devices=1)
    predict_trainer.predict(model=Model, dataloaders=predictDataloader, ckpt_path=args.ckpt_path)
              
if __name__ == '__main__':
    main()
