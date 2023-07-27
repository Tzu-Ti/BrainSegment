__author__ = 'Titi'

from pydicom import dcmread
import glob, os
import numpy as np
import nrrd
import cv2
from tqdm import tqdm

dataset_folder = "Hippo_25_openDataset/"

def get_UIDs(dcm):
    ContourImageSequence = dcm.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence
    UIDs = []
    for contour in ContourImageSequence:
        UID = contour.ReferencedSOPInstanceUID
        UIDs.append(UID)
    return UIDs

def conv_mm2px(di_ipp, di_iop, di_ps, contour_mm: list):
    """
    It converts a list of 3D points in millimeters to a list of 2D points in pixels.
    Args:
      di_ipp: Image Position Patient
      di_iop: Image Orientation Patient
      di_ps: pixel spacing
      contour_mm (list): list of x,y,z coordinates of the contour in mm
    Returns:
      The contour_px is being returned.
    """
    # yapf: disable
    matrix_im = [ [ di_iop[ 0 ] * di_ps[ 0 ], di_iop[ 3 ] * di_ps[ 1 ], np.finfo( np.float16 ).tiny, di_ipp[ 0 ] ],
                  [ di_iop[ 1 ] * di_ps[ 0 ], di_iop[ 4 ] * di_ps[ 1 ], np.finfo( np.float16 ).tiny, di_ipp[ 1 ] ],
                  [ di_iop[ 2 ] * di_ps[ 0 ], di_iop[ 5 ] * di_ps[ 1 ], np.finfo( np.float16 ).tiny, di_ipp[ 2 ] ],
                  [ 0                       , 0                       , 0                          , 1           ] ]
    # yapf: enable
    inv_matrix_im = np.linalg.inv(np.array(matrix_im))
    mm_len = len(contour_mm)
    contour_mm_ary = np.concatenate([np.array(contour_mm).reshape(mm_len // 3, 3), np.ones((mm_len // 3, 1))], 1)
    contour_px = np.rint(np.dot(inv_matrix_im, contour_mm_ary.T).T)[:, 0:2].astype(int)

    return contour_px

def main():
    patient_paths = glob.glob(os.path.join(dataset_folder, '*'))
    
    for patient in tqdm(patient_paths): # "Hippo_25_openDataset/01"
        print(patient)
        rtss_path = glob.glob(os.path.join(patient, 'RS*'))[0]        
        
        rtss = dcmread(rtss_path)
        UIDs = get_UIDs(rtss)
        # write UID order into txt
        order_path = os.path.join(patient, "UIDs.txt")
        with open(order_path, 'w') as f:
            for uid in UIDs:
                f.writelines(uid + '\n')
        
        # get label, ex. {"2": 'Hippocampus_R', "1": 'Hippocampus_L'}
        number_label_dict = {}
        RTROIObservationsSequence = rtss.RTROIObservationsSequence
        for RTROIObservations in RTROIObservationsSequence:
            number = RTROIObservations.ReferencedROINumber
            label = RTROIObservations.ROIObservationLabel
            number_label_dict[number] = label
        print(number_label_dict)
        # get contour data
        ROIContourSequence = rtss.ROIContourSequence
        for Sequence in ROIContourSequence:
            number = Sequence.ReferencedROINumber # which label
            volume = np.zeros([256, 256, len(UIDs)])

            ContourSequence = Sequence.ContourSequence
            for seq in ContourSequence:
                mr_id = seq.ContourImageSequence[0].ReferencedSOPInstanceUID
                mr_number = UIDs.index(mr_id)


                ds = dcmread(os.path.join(patient, 'MR.{}.dcm'.format(mr_id)))
                mr_info = {}
                mr_info['spacing'] = ds.PixelSpacing
                mr_info['position'] = ds.ImagePositionPatient
                mr_info['orientation'] = ds.ImageOrientationPatient
                mr_info['mrarray'] = ds.pixel_array

                # polygon contour data
                contour_points_mm = seq.ContourData
                # mm to pixel
                contour_points_px = conv_mm2px(mr_info['position'], mr_info['orientation'], mr_info['spacing'],
                                               contour_points_mm)

                mask_img = cv2.fillConvexPoly(np.zeros(mr_info["mrarray"].shape, np.uint8), contour_points_px, 1)
                volume[:, :, mr_number] = mask_img
            nrrd_path = os.path.join(patient, '{}.nrrd'.format(number_label_dict[number]))
            nrrd.write(nrrd_path, volume)
    
if __name__ == '__main__':
    main()