import numpy.random as random
import numpy as np
import tensorflow as tf
import json
import glob
import os
import random
import cv2
from cfg import cfg

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def compute_per_iou(pred, label, cls_nums, hist):
    hist += fast_hist(label.flatten(), pred.flatten(), cls_nums)
    mIoUs = per_class_iu(hist)
    mIoU = np.nanmean(mIoUs)
    return mIoUs, mIoU

def compute_iou_BDDS(pred, label, cls_nums):
    hist = np.zeros((cls_nums, cls_nums))
    hist+=fast_hist(label.flatten(), pred.flatten(), cls_nums)
    mIoUs = per_class_iu(hist)
    mIoU = np.nanmean(mIoUs)
    return mIoU