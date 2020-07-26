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
from loss_functions import *
from vis_seg import *
from net import *
from image_reader import *
from compute_mIoU import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--snapshot_dir", default='./snapshots_deeplab_based', help="path of snapshots")
parser.add_argument("--source_txt_path", default='../GTA5_data_train.txt', help="path of txt file for source domain")
parser.add_argument("--target_txt_path", default='../Cityscapes_data_train.txt',
                    help="path of txt file for target domain")
parser.add_argument("--semantic_json_path", default='../GTA5_semantic_vectors.json',
                    help="path of json file for semantic vectors")
parser.add_argument("--info_json_path", default='../GTA5_info.json', help="info path of palette")
parser.add_argument("--source_domain_name", default='GTA5', help="name of source domain")
parser.add_argument("--target_domain_name", default='Cityscapes', help="name of target domain")
parser.add_argument("--cls_nums", type=int, default=19, help="number of classes")
parser.add_argument("--vis_image_path", default='./train_vis_deeplab_based/', help="path of training visible results.")
parser.add_argument("--semantic_vector_length", type=int, default=300, help="length of semantic vector")
parser.add_argument("--checkpoint_path", default='../deeplab_resnet_renamed/', help="restore ckpt")
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam optimizer')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum term of SGD optimizer')
parser.add_argument('--power', dest='power', type=float, default=0.9, help='power of SGD optimizer')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--epoch_decay', dest='epoch_decay', type=int, default=5, help='# of epoch then decay')
parser.add_argument('--base_lr', type=float, default=0.00025, help='initial learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Regularisation parameter for L2-loss')
parser.add_argument('--total_step', dest='total_step', type=int, default=250001, help='training steps in total')
parser.add_argument("--summary_pred_every", type=int, default=50, help="times to summary.")
parser.add_argument("--cal_IoU_every", type=int, default=100, help="times to calculate mIoU.")
parser.add_argument("--show_vis_every", type=int, default=5000, help="times to save visible reaults.")
parser.add_argument("--save_pred_every", type=int, default=5000, help="times to save.")

args = parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def get_basic_variable_name(var_name):
    return var_name.split(':')[0]

def main():
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    source_file_list = get_image_name_list(args.source_txt_path)
    target_file_list = get_image_name_list(args.target_txt_path)
    source_file_length = len(source_file_list)
    target_file_length = len(target_file_list)
    total_step = args.total_step
    source_img_mean = get_mean(args.source_domain_name)
    source_kv_map, target_kv_map = get_label_kv_map(args.source_domain_name)
    words, semantic_matrix_trans_feed = get_semantic_words_and_matrix(args.semantic_json_path)
    semantic_vector_length = args.semantic_vector_length
    cls_nums = args.cls_nums
    source_image = tf.placeholder(tf.float32, shape=[1, cfg.image_height, cfg.image_width, 3], name="source_image")
    source_label = tf.placeholder(tf.float32, shape=[1, cfg.image_height, cfg.image_width, 1], name="source_label")
    semantic_matrix_trans = tf.placeholder(tf.float32, shape=[semantic_vector_length, cls_nums],
                                           name="semantic_matrix_trans")
    target_image = tf.placeholder(tf.float32, shape=[1, cfg.image_height, cfg.image_width, 3], name="target_image")

    source_net = DeepLabResNetModel({'data': source_image}, is_training=True, reuse=False, num_classes=cls_nums)
    source_feature_block4 = source_net.layers['res5c']
    source_feature_block3 = source_net.layers['res4b22']
    source_feature_block2 = source_net.layers['res3b3']

    target_net = DeepLabResNetModel({'data': target_image}, is_training=True, reuse=True, num_classes=cls_nums)
    target_feature_block4 = target_net.layers['res5c']
    target_feature_block3 = target_net.layers['res4b22']
    target_feature_block2 = target_net.layers['res3b3']

    source_semantic_restore_block2, source_ASPP_block2 = transfer_network(source_feature_block2, semantic_vector_length,
                                                                          cls_nums, reuse=False,
                                                                          name="transfer_network_block2")
    target_semantic_restore_block2, target_ASPP_block2 = transfer_network(target_feature_block2, semantic_vector_length,
                                                                          cls_nums, reuse=True,
                                                                          name="transfer_network_block2")

    source_semantic_restore_block3, source_ASPP_block3 = transfer_network(source_feature_block3, semantic_vector_length,
                                                                          cls_nums, reuse=False,
                                                                          name="transfer_network_block3")
    target_semantic_restore_block3, target_ASPP_block3 = transfer_network(target_feature_block3, semantic_vector_length,
                                                                          cls_nums, reuse=True,
                                                                          name="transfer_network_block3")

    source_semantic_restore_block4, source_ASPP_block4 = transfer_network(source_feature_block4, semantic_vector_length,
                                                                          cls_nums, reuse=False,
                                                                          name="transfer_network_block4")
    target_semantic_restore_block4, target_ASPP_block4 = transfer_network(target_feature_block4, semantic_vector_length,
                                                                          cls_nums, reuse=True,
                                                                          name="transfer_network_block4")

    target_out = tf.image.resize_bilinear(target_ASPP_block4, tf.shape(target_image)[1:3, ])
    target_seg_raw = tf.argmax(target_out, dimension=3)

    # compute_consine_loss

    block2_cosine_loss = (cosine_loss(source_semantic_restore_block2, semantic_matrix_trans) + cosine_loss(
        target_semantic_restore_block2, semantic_matrix_trans) + cosine_loss(source_semantic_restore_block2,
                                                                             target_semantic_restore_block2)) / 3

    block3_cosine_loss = (cosine_loss(source_semantic_restore_block3,
                                      semantic_matrix_trans) + cosine_loss(target_semantic_restore_block3,
                                                                           semantic_matrix_trans) + cosine_loss(
        source_semantic_restore_block3, target_semantic_restore_block3)) / 3
    block4_cosine_loss = (cosine_loss(source_semantic_restore_block4,
                                      semantic_matrix_trans) + cosine_loss(target_semantic_restore_block4,
                                                                           semantic_matrix_trans) + cosine_loss(
        source_semantic_restore_block4, target_semantic_restore_block4)) / 3
    restore_loss = 0.25 * block2_cosine_loss + 0.5 * block3_cosine_loss + block4_cosine_loss

    # compute_segmentation_loss
    reshaped_source_label = tf.image.resize_nearest_neighbor(source_label,
                                                             tf.stack(source_ASPP_block4.get_shape()[1:3]))
    squeezed_source_image_label = tf.squeeze(reshaped_source_label, squeeze_dims=[3])
    raw_onehot_gt = tf.reshape(squeezed_source_image_label, [-1, ])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_onehot_gt, args.cls_nums - 1)), 1)
    onehot_gt = tf.cast(tf.gather(raw_onehot_gt, indices), tf.int32)
    source_raw_onehot_prediction_block4 = tf.reshape(source_ASPP_block4, [-1, cls_nums])
    source_onehot_prediction_block4 = tf.gather(source_raw_onehot_prediction_block4, indices)
    semantic_segmentation_loss_block4 = softmax_cross_entropy_loss(source_onehot_prediction_block4, onehot_gt)
    source_raw_onehot_prediction_block3 = tf.reshape(source_ASPP_block3, [-1, cls_nums])
    source_onehot_prediction_block3 = tf.gather(source_raw_onehot_prediction_block3, indices)
    semantic_segmentation_loss_block3 = softmax_cross_entropy_loss(source_onehot_prediction_block3, onehot_gt)
    semantic_segmentation_loss = 0.1 * semantic_segmentation_loss_block3 + semantic_segmentation_loss_block4

    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'transfer_network' in v.name and 'weights' in v.name]

    # total_loss
    total_loss = cfg.restore_lambda * restore_loss + semantic_segmentation_loss + tf.add_n(l2_losses)

    # summaries
    restore_loss_sum = tf.summary.scalar("restore_loss", restore_loss)
    semantic_segmentation_loss_sum = tf.summary.scalar("semantic_segmentation_loss", semantic_segmentation_loss)
    total_loss_sum = tf.summary.scalar("total_loss", total_loss)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    restore_vars = [v for v in tf.global_variables() if 'transfer_network' not in v.name]

    SGD_step = tf.placeholder(tf.float32, None, name='SGD_learning_step')
    SGD_lr = tf.scalar_mul(args.base_lr, tf.pow((1 - SGD_step / total_step), args.power))
    backbone_optim = tf.train.MomentumOptimizer(SGD_lr, args.momentum)
    transfer_optim = tf.train.MomentumOptimizer(SGD_lr*10, args.momentum)

    backbone_vars = [v for v in tf.trainable_variables() if 'transfer_network' not in v.name]
    transfer_vars = [v for v in tf.trainable_variables() if 'transfer_network' in v.name]
    backbone_params_grads_and_vars = backbone_optim.compute_gradients(total_loss, var_list=backbone_vars)
    backbone_params_train = backbone_optim.apply_gradients(backbone_params_grads_and_vars)
    transfer_params_grads_and_vars = transfer_optim.compute_gradients(total_loss, var_list=transfer_vars)
    transfer_params_train = transfer_optim.apply_gradients(transfer_params_grads_and_vars)
    params_train = tf.group(backbone_params_train, transfer_params_train)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)

    loader = tf.train.Saver(var_list=restore_vars)
    checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)
    load(loader, sess, checkpoint)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    shuffle(source_file_list)
    shuffle(target_file_list)
    hist = np.zeros((cls_nums, cls_nums))
    sess.graph.finalize()
    for step in range(total_step):
        source_image_batch, source_image_name_batch, source_label_batch, target_image_batch, target_image_name_batch, target_label_batch = train_image_reader(
            source_file_list, target_file_list, (cfg.image_width, cfg.image_height),
            (cfg.image_width, cfg.image_height),
            args.source_domain_name, args.target_domain_name, source_kv_map, target_kv_map, step, source_img_mean)
        source_image_batch = np.expand_dims(np.array(source_image_batch).astype(np.float32), axis=0)
        source_label_batch = np.expand_dims(np.array(source_label_batch).astype(np.float32), axis=0)
        source_label_batch = np.expand_dims(np.array(source_label_batch).astype(np.float32), axis=-1)
        target_image_batch = np.expand_dims(np.array(target_image_batch).astype(np.float32), axis=0)

        feed_dict = {source_image: source_image_batch, source_label: source_label_batch,
                     target_image: target_image_batch, semantic_matrix_trans: semantic_matrix_trans_feed,
                     SGD_step: step}
        restore_loss_value, semantic_segmentation_loss_value, total_loss_value, _ = sess.run(
            [restore_loss, semantic_segmentation_loss, total_loss, params_train], feed_dict=feed_dict)

        print("step =: ", step)
        print("restore_loss_value =: ", restore_loss_value)
        print("semantic_segmentation_loss_value := ", semantic_segmentation_loss_value)
        print("total_loss_value := ", total_loss_value)
        print(".............................................................................")

        if step % source_file_length == 0:
            shuffle(source_file_list)
        if step % target_file_length == 0:
            shuffle(target_file_list)

        if step % args.summary_pred_every == 0:
            restore_loss_sum_value, semantic_segmentation_loss_sum_value, total_loss_sum_value = sess.run(
                [restore_loss_sum, semantic_segmentation_loss_sum, total_loss_sum],
                feed_dict=feed_dict)
            summary_writer.add_summary(restore_loss_sum_value, step)
            summary_writer.add_summary(semantic_segmentation_loss_sum_value, step)
            summary_writer.add_summary(total_loss_sum_value, step)

        if step % args.cal_IoU_every == 0:
            target_seg = sess.run(target_seg_raw, feed_dict=feed_dict)
            mIoUs, mIoU = compute_per_iou(target_seg[0], np.array(target_label_batch, dtype=np.int32), cls_nums,
                                          hist)
            print("----------------------------------------------------------")
            print(mIoUs)
            print(mIoU)
            print("----------------------------------------------------------")
        if step % args.show_vis_every == 0:
            target_seg = sess.run(target_seg_raw, feed_dict=feed_dict)
            get_vis_images(target_image_batch[0], target_image_name_batch, target_seg[0], target_label_batch,
                           cls_nums, args.info_json_path, source_img_mean, args.vis_image_path)
        if step % args.save_pred_every == 0:
            save(saver, sess, args.snapshot_dir, step)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
