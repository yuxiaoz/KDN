import numpy.random as random
import numpy as np
import tensorflow as tf
import json
import glob
import os
import random
import cv2
from PIL import Image
from utils import *

def get_palette(json_path):
    json_file = open(json_path, encoding='utf-8')
    palette = json.load(json_file)["palette"]
    return palette

def get_coloured_pred(pred, palette, cls_nums):
    palette = np.array(palette,dtype = np.float32)
    pred = np.array(pred).astype(np.int32)
    rgb_pred = np.zeros(shape=[pred.shape[0], pred.shape[1], 3],dtype=np.float32)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            k =pred[i][j]
            if  k< cls_nums:
                rgb_pred[i][j] = palette[k][::-1]
    return rgb_pred

def get_vis_images(image, image_name, pred, gt, cls_nums, json_path, mean, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_bgr = np.array(image).astype(np.float32) + np.array(mean).astype(np.float32)
    palette = get_palette(json_path)
    coloured_pred = get_coloured_pred(pred, palette, cls_nums)
    coloured_gt = get_coloured_pred(gt, palette, cls_nums)
    img_format = ".png"
    save_image_name = save_path + image_name + img_format
    save_pred_name = save_path + image_name + "_pred" + img_format
    save_gt_name = save_path + image_name + "_gt" + img_format
    cv2.imwrite(save_image_name, img_bgr)
    cv2.imwrite(save_pred_name, coloured_pred)
    cv2.imwrite(save_gt_name, coloured_gt)