# src/visualization.py
import os
import pickle
import numpy as np
from tifffile import imsave
import nibabel as nib


def export_tiff_predictions(predictions, id_list, out_dir="./channel_split/"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, ID in enumerate(id_list):
        pred = predictions[i]
        # Apply thresholding to segmentation masks
        if pred.max() > 0:
            pred = (pred > 0.5).astype(np.uint8) * 255
        imsave(os.path.join(out_dir, f"{ID}_prediction.tif"), pred)
        print(f"Saved: {out_dir}{ID}_prediction.tif")


def export_ground_truth(id_list, out_dir="./channel_split/"):
    for ID in id_list:
        # Assuming files exist: _flair.nii.gz, _t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz, _seg.nii.gz
        img_files = [f'./data/{ID}_flair.nii.gz', f'./data/{ID}_t1.nii.gz',
                     f'./data/{ID}_t1ce.nii.gz', f'./data/{ID}_t2.nii.gz', f'./data/{ID}_seg.nii.gz']
        images = [np.array(nib.load(f).dataobj) for f in img_files]
        stacked = np.stack(images, axis=0)
        # Save each modality and segmentation as TIFF
        for idx, arr in enumerate(stacked):
            arr_uint8 = (arr / arr.max() * 255).astype(np.uint8)
            imsave(os.path.join(
                out_dir, f"{ID}_modality{idx+1}.tif"), arr_uint8)
        print(f"Saved ground truth TIFFs for {ID}")


def threshold_and_save(predictions, id_list, out_dir="./channel_split/"):
    for i, ID in enumerate(id_list):
        pred = predictions[i]
        pred_bin = (pred > 0.5).astype(np.uint8) * 255
        imsave(os.path.join(out_dir, f"{ID}_prediction_bin.tif"), pred_bin)
        print(f"Saved: {out_dir}{ID}_prediction_bin.tif")
