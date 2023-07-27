import glob, os
import nrrd
import argparse
import numpy as np
from tqdm import tqdm
import json

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', default='Hippo_dataset_VGHTC_share')
    parser.add_argument('--patient_path')
    parser.add_argument('--map_label', action="store_true")

    return parser.parse_args()

def IntensityClipping(x):
    std = x.std()
    mean = x.mean()
    MAX = mean + 3 * std
    MIN = mean - 3 * std
    x = np.clip(x, MIN, MAX)
    return x

def Mapping(seg, labelmap):
    new = seg.copy()
    for k in labelmap.keys():
        label = labelmap[k]
        loc = np.where(seg==int(k))
        if len(loc) == 1:
            continue
        else:
            X, Y, Z = loc
        for x, y, z in zip(X, Y, Z):
            new[x, y, z] = label
    return new

def main():
    args = parse()
    if args.patient_path:
        patient_paths = [args.patient_path]
    else:
        patient_paths = glob.glob(os.path.join(args.folder_path, 'HIP*'))

    for patient_path in tqdm(patient_paths):
        mr, header = nrrd.read(os.path.join(patient_path, 'MR.nrrd'))
        seg, header = nrrd.read(os.path.join(patient_path, 'Segmentation.seg.nrrd'))

        mr = IntensityClipping(mr)
        if args.map_label:
            with open(os.path.join(args.folder_path, 'LabelMap.json')) as f:
                LabelMap = json.load(f)
            seg = Mapping(seg, LabelMap)
        
        mr_npy_folder = os.path.join(patient_path, 'MR_npy')
        seg_npy_folder = os.path.join(patient_path, 'Seg_npy')
        if args.map_label:
            seg_npy_folder = os.path.join(patient_path, 'mapped_Seg_npy')
        if not os.path.isdir(mr_npy_folder):
            os.makedirs(mr_npy_folder)
        if not os.path.isdir(seg_npy_folder):
            os.makedirs(seg_npy_folder)

        assert mr.shape == seg.shape
        h, w, d = mr.shape
        for idx in range(d):
            m = mr[:, :, idx]
            s = seg[:, :, idx]
            np.save(os.path.join(mr_npy_folder, '{:03d}.npy'.format(idx)), m)
            np.save(os.path.join(seg_npy_folder, '{:03d}.npy'.format(idx)), s)

if __name__ == '__main__':
    main()