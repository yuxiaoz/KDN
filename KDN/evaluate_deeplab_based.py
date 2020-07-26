import numpy as np
import tensorflow as tf
import json
import glob
import os
import random
import cv2
import argparse
from random import shuffle
import time
import math
from cfg import cfg

from deeplab_v2 import *
from vis_seg import *
from net import *
from image_reader import *
from compute_mIoU import *

parser = argparse.ArgumentParser(description='')
parser.add_argument("--snapshot_dir", default='./snapshots_deeplab_based_GTA5', help="path of snapshots")
parser.add_argument("--write_path", default='./mIoUs_deeplab_based_GTA5.txt', help="path of mIoU txt")
parser.add_argument("--target_txt_path", default='../Cityscapes_data_val.txt',
                    help="path of txt file for target domain")
parser.add_argument("--info_json_path", default='../GTA5_info.json', help="info path of palette")
parser.add_argument("--source_domain_name", default='GTA5', help="name of source domain")
parser.add_argument("--target_domain_name", default='Cityscapes', help="name of target domain")
parser.add_argument("--output_path", default='./test_deeplab_based_GTA5/', help="path of test visible results.")
parser.add_argument("--semantic_vector_length", type=int, default=300, help="length of semantic vector")
parser.add_argument("--cls_nums", type=int, default=19, help="number of classes")

args = parser.parse_args()

def main():
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    target_file_list = get_image_name_list(args.target_txt_path)
    source_img_mean = get_mean(args.source_domain_name)
    cls_nums = args.cls_nums
    semantic_vector_length = args.semantic_vector_length
    source_kv_map, target_kv_map = get_label_kv_map(args.source_domain_name)
    target_image = tf.placeholder(tf.float32, shape=[1, cfg.image_height, cfg.image_width, 3], name="target_image")

    target_net = DeepLabResNetModel({'data': target_image}, is_training=True, reuse=False, num_classes=cls_nums)
    target_feature_block4 = target_net.layers['res5c']

    target_semantic_restore_block4, target_ASPP_block4 = transfer_network(target_feature_block4, semantic_vector_length,
                                                                          cls_nums, reuse=False,
                                                                          name="transfer_network_block4")

    target_out = tf.image.resize_bilinear(target_ASPP_block4, tf.shape(target_image)[1:3, ])
    target_seg_raw = tf.argmax(target_out, dimension=3)

    restore_var = [v for v in tf.global_variables()]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)
    checkpoint = tf.train.latest_checkpoint(args.snapshot_dir)
    saver.restore(sess, checkpoint)

    counter = 0
    target_file_length = len(target_file_list)
    shuffle(target_file_list)
    hist = np.zeros((cls_nums, cls_nums))
    sess.graph.finalize()

    for step in range(target_file_length):
        target_image_batch, target_image_name_batch, target_label_batch = test_image_reader(target_file_list, (
        cfg.image_width, cfg.image_height), args.target_domain_name, target_kv_map, step, source_img_mean)
        target_image_batch = np.expand_dims(np.array(target_image_batch).astype(np.float32), axis=0)
        feed_dict = {target_image: target_image_batch}
        target_seg = sess.run(target_seg_raw, feed_dict=feed_dict)
        mIoUs, mIoU = compute_per_iou(target_seg[0], np.array(target_label_batch, dtype=np.int32), cls_nums,
                                      hist)
        """
        get_vis_images(target_image_batch[0], target_image_name_batch, target_seg[0], target_label_batch,
                       cls_nums, args.info_json_path, source_img_mean, args.output_path)
        """
        counter += 1
        print(counter)

    print("----------------------------------------------------------")
    print(mIoUs)
    print(mIoU)
    print("----------------------------------------------------------")
    write_file = open(args.write_path, "a+", encoding='UTF-8')
    write_file.write("mIoUs:" + "\n")
    write_file.write(str(mIoUs) + "\n")
    write_file.write("mIoU:" + "\n")
    write_file.write(str(mIoU))
    write_file.close()
    print("done!")


if __name__ == '__main__':
    main()