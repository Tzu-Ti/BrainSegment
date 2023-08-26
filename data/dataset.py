import torch
import torchvision
from torch.utils.data.dataset import Dataset
import os, glob
from pydicom import dcmread
import numpy as np
import nrrd
import random
from torchvision.transforms import functional as TF
from torch.nn import functional as F

def open_txt(path):
    with open(path, 'r') as f:
        lst = [line.strip() for line in f.readlines()]
    return lst

class RotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class HippoDataset(Dataset):
    def __init__(self, folder='../Hippo_25_openDataset', MaskList_path='trainMask.txt', train=True):
        self.folder = folder
        self.train = train
        
        MaskList = open_txt(MaskList_path)
        
        self.List = MaskList
        
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
        self.transform = torchvision.transforms.Compose(transforms)
        
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

        # data augmentation rotate
        if self.train:
            angle = random.choice([0, 90, 180, 270])
            mask = TF.rotate(mask, angle)
            img = TF.rotate(img, angle)
        
        return img, mask
    
    def __len__(self):
        return len(self.List)
    
class VGHTCDataset(Dataset):
    def __init__(self, folder='../Hippo_dataset_VGHTC_share', lst='../Hippo_dataset_VGHTC_share/trainList.txt', size=256, num_classes=44, patient_name='HIP_002', train=True):
        self.folder = folder
        self.train = train
        self.num_classes = num_classes

        if not train and patient_name:
            patient_paths = [os.path.join(folder, patient_name)]
        else:
            name_lst = open_txt(lst)
            patient_paths = [os.path.join(folder, name) for name in name_lst]

        self.all_mr_path = []
        for patient_path in patient_paths:
            mrs = glob.glob(os.path.join(patient_path, 'MR_npy', '*.npy'))
            self.all_mr_path += mrs     
        self.all_seg_path = [path.replace('MR_npy', 'mapped_Seg_npy') for path in self.all_mr_path]
        
        if not train and patient_name:
            self.all_mr_path.sort()
            self.all_seg_path.sort()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
            torchvision.transforms.Resize([size, size], antialias=True)
        ])
        self.Resize = torchvision.transforms.Resize([size, size], torchvision.transforms.InterpolationMode.NEAREST, antialias=True)

    def __getitem__(self, index):
        mr_path = self.all_mr_path[index]
        seg_path = self.all_seg_path[index]

        img = np.load(mr_path)
        seg = np.load(seg_path)

        # Image Normalization
        img = (img - img.min()) / (img.max() - img.min())
        img = self.transform(img).type(torch.float32)

        # Segmentation class to one hot
        seg = torch.from_numpy(seg).type(torch.long)
        seg = F.one_hot(seg, num_classes=self.num_classes)
        seg = seg.permute(2, 0, 1)
        seg = self.Resize(seg)

        # data augmentation rotate
        if self.train:
            angle = random.choice([0, 90, 180, 270])
            seg = TF.rotate(seg, angle)
            img = TF.rotate(img, angle)

        return img, seg

    def __len__(self):
        return len(self.all_mr_path)
    
class VGHTCDatasetNoSeg(Dataset):
    def __init__(self, folder='../Hippo_dataset_VGHTC_share', size=256, num_classes=44, patient_name='HIP_002'):
        self.folder = folder
        self.num_classes = num_classes

        patient_paths = [os.path.join(folder, patient_name)]

        self.all_mr_path = []
        for patient_path in patient_paths:
            mrs = glob.glob(os.path.join(patient_path, 'MR_npy', '*.npy'))
            self.all_mr_path += mrs     

        self.all_mr_path.sort()


        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
            torchvision.transforms.Resize([size, size], antialias=True)
        ])
        self.Resize = torchvision.transforms.Resize([size, size], torchvision.transforms.InterpolationMode.NEAREST, antialias=True)

    def __getitem__(self, index):
        mr_path = self.all_mr_path[index]

        img = np.load(mr_path)

        # Image Normalization
        img = (img - img.min()) / (img.max() - img.min())
        img = self.transform(img).type(torch.float32)

        return img

    def __len__(self):
        return len(self.all_mr_path)
        
if __name__ == '__main__':
    # D = HippoDataset()
    D = VGHTCDatasetNoSeg(patient_name='HIP_002')
    for img, seg in D:
        pass
        # print(img.shape)
        # print(img.dtype)
        # print(seg.shape)