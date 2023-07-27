import torch
import torchvision
from torch.utils.data.dataset import Dataset
import os, glob
from pydicom import dcmread
import numpy as np
import nrrd

def open_txt(path):
    with open(path, 'r') as f:
        lst = [line.strip() for line in f.readlines()]
    return lst

class HippoDataset(Dataset):
    def __init__(self, folder='../Hippo_25_openDataset', MaskList_path='trainMask.txt', noMaskList_path='trainNoMask.txt', train=True):
        self.folder = folder
        
        MaskList = open_txt(MaskList_path)
        NoMaskList = open_txt(noMaskList_path)
        
#         self.trainList = trainMaskList + trainNoMaskList[:len(trainMaskList)]
        if train:
            self.List = MaskList
        else:
            self.List = MaskList + NoMaskList
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        
        self.totensor = torchvision.transforms.ToTensor()
        
    def get_mask(self, path, mr_number):
        data, header = nrrd.read(path)
        mask = data[:, :, mr_number].astype(np.float32)
        mask = self.totensor(mask)
        return mask
        
    def __getitem__(self, index):
        mr_path = self.List[index]
        patient_id, mr_id = mr_path.split('/')
        
        # get MR image
        mr_path = os.path.join(self.folder, patient_id, "MR.{}.dcm".format(mr_id))
        ds = dcmread(mr_path)
        img = ds.pixel_array
        img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
        img = self.transform(img)
        
        # get UID order list
        UIDs_path = os.path.join(self.folder, patient_id, 'UIDs.txt')
        UIDs = open_txt(UIDs_path)
        
        # get MR image order
        mr_number = UIDs.index(mr_id)
        
        # get label nrrd
        left3Dnrrd_path = os.path.join(self.folder, patient_id, "Hippocampus_L.nrrd")
        right3Dnrrd_path = os.path.join(self.folder, patient_id, "Hippocampus_R.nrrd")
        left_mask = self.get_mask(left3Dnrrd_path, mr_number)
        right_mask = self.get_mask(right3Dnrrd_path, mr_number)
        
        mask = torch.cat([left_mask, right_mask], dim=0).type(torch.long)
        
        return img, mask
    
    def __len__(self):
        return len(self.List)
        
if __name__ == '__main__':
    D = HippoDataset()
    for _ in D:
        break