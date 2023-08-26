# Brain Segmentation on VGHTC dataset
## Preprocess
### 1. .nrrd Correction
- Correct the segmentation to make the coordinates and direction the same as MR.
- Generate new 3D MR **"MR.nrrd"** and new 3D segmentation **"Segmentation.seg.nrrd"** in which folder.
```
# for whole folder
$ python3 nrrd_correction.py --folder_path Hippo_dataset_VGHTC_share
# for only one patient
$ python3 nrrd_correction.py --patient_path Hippo_dataset_VGHTC_share/test
```

### 2. Convert 3D .nrrd to 2D .npy
- Rearrange the label numbers
- Create two folders in which folder
  - **"mapped_seg_npy"**: a set of ndarray of 2D segmentation
  - **"MR_npy"**: a set of ndarray of 2D MR
```
# for whole folder
$ python3 nrrd2npy.py --folder_path Hippo_dataset_VGHTC_share --map_label
# for only one patient
$ python3 nrrd2npy.py --patient_path Hippo_dataset_VGHTC_share/test
```

### 3. Create label map
- Because origin label is not continuous, rearrange 0 to 44.
- Already wrote in **"LabelMap.json"**

## Segment
- Output will save in **"Output/patient_name"**
```
$ python3 train_VGHTC.py --model_name 0722VGHTC --predict --ckpt_path lightning_logs/0722VGHTC/checkpoints/epoch\=499-step\=374000.ckpt --patient_name test --save_result
```

## Training
```
$ python3 train_VGHTC.py --model_name Test --train
```