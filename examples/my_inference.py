#!/usr/bin/python
"""
    Adapted from the original C++ example: densecrf/examples/dense_inference.cpp
    http://www.philkr.net/home/densecrf Version 2.2
"""

import os
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # ---- 1. set your image path with '.jpg' format ----
    img_dir = '/media/nercms/NERCMS/GepengJi/CRF/CRF/DPFan/DAVIS-test'
    # ---- 2. set your prediction path with '.png' format ----
    pred_dir = '/media/nercms/NERCMS/GepengJi/CRF/Gepeng_ECCV/submit_DAVIS/DAVIS_20'
    # ---- 3. set your output path ----
    save_dir = '/media/nercms/NERCMS/GepengJi/CRF/Gepeng_ECCV/submit_DAVIS/DAVIS_20_crf'

    EPSILON = 1e-8
    for seq_name in os.listdir(pred_dir):
        img_seq_dir = os.path.join(img_dir, seq_name, 'Imgs')
        pred_seq_dir = os.path.join(pred_dir, seq_name)
        save_seq_dir = os.path.join(save_dir, seq_name)
        os.makedirs(save_seq_dir, exist_ok=True)

        for img_name in os.listdir(pred_seq_dir):
            print(os.path.join(img_seq_dir, img_name.replace('.png', '.jpg')))
            img = cv2.imread(os.path.join(img_seq_dir, img_name.replace('.png', '.jpg')), 1)
            annos = cv2.imread(os.path.join(pred_seq_dir, img_name), 0)

            labels = relabel_sequential(cv2.imread(os.path.join(pred_seq_dir, img_name), 0))[0].flatten()
            output = os.path.join(save_seq_dir, img_name)
            # ---- salient or not ----
            M = 2
            tau = 1.05
            # ---- Setup the CRF model ----
            d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

            anno_norm = annos / 255.
            n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
            p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

            U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
            U[0, :] = n_energy.flatten()
            U[1, :] = p_energy.flatten()

            d.setUnaryEnergy(U)

            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

            # ---- Do the inference ----
            infer = np.array(d.inference(1)).astype('float32')
            res = infer[1, :]

            # res *= 255 / res.max()
            res = res * 255
            res = res.reshape(img.shape[:2])
            # NOTES: please uncomment it if you want to binarized the post-processed mask `res`
            # ret, res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)  # 220 241
            cv2.imwrite(output, res.astype('uint8'))
