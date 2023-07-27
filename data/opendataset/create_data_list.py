__author__ = 'Titi'

from pydicom import dcmread
import glob, os
import numpy as np
import nrrd
import cv2
from tqdm import tqdm

dataset_folder = "../../Hippo_25_openDataset/"

def write_txt(name, lst):
    with open(name, 'w') as f:
        for l in lst:
            f.writelines(l + '\n')

def check_mask(nrrd_path):
    readdata, header = nrrd.read(nrrd_path)
    mask_lst = []
    for mask in readdata.transpose(2, 0, 1):
        mask_lst.append(np.max(mask))
    return mask_lst

def split_train_test():
    patient_paths = glob.glob(os.path.join(dataset_folder, '*'))
    
    with_mask = []
    no_mask = []
    
    for patient in tqdm(patient_paths):
        patient_id = os.path.basename(patient)

        left_nrrd_path = os.path.join(patient, "Hippocampus_L.nrrd")
        right_nrrd_path = os.path.join(patient, "Hippocampus_R.nrrd")
        UIDs_path = os.path.join(patient, "UIDs.txt")
        with open(UIDs_path, 'r') as f:
            UIDs = [line.strip() for line in f.readlines()]
            
        L_lst = check_mask(left_nrrd_path)
        R_lst = check_mask(right_nrrd_path)
        for index, (L, R) in enumerate(zip(L_lst, R_lst)):
            mr_path = os.path.join(patient_id, UIDs[index])
            if L == 1.0 or R == 1.0:
                with_mask.append(mr_path)
            elif L == 0.0 and R == 0.0:
                no_mask.append(mr_path)

    with_mask_length = len(with_mask)
    no_mask_length = len(no_mask)
    print("Total mask:")
    print("With mask: {}, no mask: {}".format(with_mask_length, no_mask_length))
    
    # shuffle
    np.random.shuffle(no_mask)
    np.random.shuffle(with_mask)
    
    write_txt('trainMask.txt', with_mask[:int(with_mask_length*0.8)])
    write_txt('trainNoMask.txt', no_mask[:int(no_mask_length*0.8)])
    write_txt('testMask.txt', with_mask[int(with_mask_length*0.8):])
    write_txt('testNoMask.txt', no_mask[int(no_mask_length*0.8):])

def create_one_patient_list():
    patient_paths = glob.glob(os.path.join(dataset_folder, '*'))
    for patient_path in patient_paths:
        UIDs_path = os.path.join(patient_path, "UIDs.txt")
        with open(UIDs_path, 'r') as f:
            UIDs = [line.strip() for line in f.readlines()]
        print("Number of MRs:", len(UIDs))

        patient_id = os.path.basename(patient_path)
        lst = []
        for uid in UIDs:
            mr_path = os.path.join(patient_id, uid)
            lst.append(mr_path)
        
        write_txt('{}.txt'.format(patient_id), lst)
if __name__ == '__main__':
    create_one_patient_list()