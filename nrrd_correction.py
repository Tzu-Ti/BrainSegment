import SimpleITK as sitk
import nrrd
import numpy as np
import argparse
import glob, os
from tqdm import tqdm

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', default='Hippo_dataset_VGHTC_share')
    parser.add_argument('--patient_path')

    return parser.parse_args()

def resample(img1, img2): # resample img2 to match img1
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetReferenceImage(img1)
    resampled_image = resampler.Execute(img2)

    resampled_image.SetDirection(img1.GetDirection())
    resampled_image.SetOrigin(img1.GetOrigin())
    return resampled_image

def main():
    args = parse()
    if args.patient_path:
        patient_paths = [args.patient_path]
    else:
        patient_paths = glob.glob(os.path.join(args.folder_path, 'HIP*'))
    
    for patient_path in tqdm(patient_paths):
        mr_path = glob.glob(os.path.join(patient_path, '*MPRAGE*.nrrd'))[0]
        seg_path = glob.glob(os.path.join(patient_path, '*.seg.nrrd'))[0]
        
        mr = sitk.ReadImage(mr_path)
        seg = sitk.ReadImage(seg_path)

        new_seg = resample(mr, seg)

        sitk.WriteImage(mr, os.path.join(patient_path, 'MR.nrrd'))
        sitk.WriteImage(new_seg, os.path.join(patient_path, 'Segmentation.seg.nrrd'))

if __name__ == '__main__':
    main()