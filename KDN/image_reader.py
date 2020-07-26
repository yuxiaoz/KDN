import numpy as np
import tensorflow as tf
import json
import glob
import os
import random
import cv2

from cfg import cfg
from utils import *
from image_io_and_process import *

#read images & labels

def read_label(domain_name, kv_map, label_path, label_size):
    if domain_name == "GTA5":
        label = read_GTA5_label(label_path, kv_map, label_size)
    elif domain_name == "SYNTHIA":
        label = read_SYNTHIA_label(label_path, kv_map, label_size)
    elif domain_name == "Cityscapes":
        label = read_Cityscapes_label(label_path, kv_map, label_size)
    else:
        label = read_BDDS_label(label_path, kv_map, label_size)
    return label

def train_image_reader(source_image_name_list, target_image_name_list, source_image_size, target_image_size, source_domain_name, target_domain_name, source_kv_map, target_kv_map, step, mean):
    source_image_nums = len(source_image_name_list)
    target_iamge_nums = len(target_image_name_list)
    source_image_index = step % source_image_nums
    target_image_index = step % target_iamge_nums
    source_pair_content = source_image_name_list[source_image_index]
    target_pair_content = target_image_name_list[target_image_index]
    source_image_path = source_pair_content.split(' ')[0]
    source_label_path = source_pair_content.split(' ')[1]
    target_image_path = target_pair_content.split(' ')[0]
    target_label_path = target_pair_content.split(' ')[1]
    source_image = read_image(source_image_path, source_image_size, mean)
    target_image = read_image(target_image_path, target_image_size, mean)
    source_image_name, _ = os.path.splitext(os.path.basename(source_image_path))
    target_image_name, _ = os.path.splitext(os.path.basename(target_image_path))
    source_label = read_label(source_domain_name, source_kv_map, source_label_path, source_image_size)
    target_label = read_label(target_domain_name, target_kv_map, target_label_path, target_image_size) #to calculate metrics
    source_mirror_flag = np.random.uniform()
    if np.less(source_mirror_flag, 0.5):
        source_image = source_image[:,::-1,:]
        source_label = source_label[:, ::-1]
    target_mirror_flag = np.random.uniform()
    if np.less(target_mirror_flag, 0.5):
        target_image = target_image[:, ::-1, :]
        target_label = target_label[:, ::-1]
    return source_image, source_image_name, source_label, target_image, target_image_name, target_label

def test_image_reader(target_image_name_list, target_image_size, target_domain_name, target_kv_map, step, mean):
    target_iamge_nums = len(target_image_name_list)
    target_image_index = step % target_iamge_nums
    target_pair_content = target_image_name_list[target_image_index]
    target_image_path = target_pair_content.split(' ')[0]
    target_label_path = target_pair_content.split(' ')[1]
    target_image = read_image(target_image_path, target_image_size, mean)
    target_image_name, _ = os.path.splitext(os.path.basename(target_image_path))
    target_label = read_label(target_domain_name, target_kv_map, target_label_path, target_image_size)
    return target_image, target_image_name, target_label