import sys
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
from utils import *
slim = tf.contrib.slim

def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)

def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output

def depthwise_conv2d(input_, channel_multiplier, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, channel_multiplier])
        output = tf.nn.depthwise_conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name = 'biases', shape=[input_dim * channel_multiplier])
            output = tf.nn.bias_add(output, biases)
        return output

def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output

def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output

def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output

def batch_norm_slim(input_, is_training, activation_fn=None, name="batch_norm", scale=True):
    with tf.variable_scope(name) as scope:
        output = slim.batch_norm(
            input_,
            activation_fn=activation_fn,
            is_training=is_training,
            updates_collections=None,
            scale=scale,
            scope=scope)
        return output

def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def pyramid_pooling(input_, output_dim, kernel_size, stride = 1, padding = "SAME", name = "pyramid"):
    input_dim = input_.get_shape()[-1]
    avg_out = avg_pooling(input_ = input_, kernel_size = kernel_size, stride = kernel_size, name = name + "pa")
    conv1 = conv2d(input_ = avg_out, output_dim = output_dim, kernel_size = 1, stride = 1,  padding = "SAME", name = name + "py_conv2d", biased = False)
    output = tf.image.resize_bilinear(conv1, tf.shape(input_)[1:3,])
    return output

def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)

def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)

def residule_block_131(input_, output_dim1, output_dim2, stride = 1, dilation = 2, atrous = True, name = "res"):
    conv2dc0 = conv2d(input_ = input_, output_dim = output_dim1, kernel_size = 1, stride = stride, name = (name + '_c0'))
    conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
    conv2dc1 = conv2d(input_ = input_, output_dim = output_dim2, kernel_size = 1, stride = stride, name = (name + '_c1'))
    conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    conv2dc1_relu = relu(input_ = conv2dc1_norm)
    if atrous:
        conv2dc2 = atrous_conv2d(input_ = conv2dc1_relu, output_dim = output_dim2, kernel_size = 3, dilation = dilation, name = (name + '_c2'))
    else:
        conv2dc2 = conv2d(input_ = conv2dc1_relu, output_dim = output_dim2, kernel_size = 3, stride = stride, name = (name + '_c2'))
    conv2dc2_norm = batch_norm(input_ = conv2dc2, name = (name + '_bn2'))
    conv2dc2_relu = relu(input_ = conv2dc2_norm)
    conv2dc3 = conv2d(input_ = conv2dc2_relu, output_dim = output_dim1, kernel_size = 1, stride = stride, name = (name + '_c3'))
    conv2dc3_norm = batch_norm(input_ = conv2dc3, name = (name + '_bn3'))
    add_raw = conv2dc0_norm + conv2dc3_norm
    output = relu(input_ = add_raw)
    return output

def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    else:
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    add_raw = input_ + conv2dc1_norm
    output = relu(input_ = add_raw)
    return output

def flatten_decoded_tensor(input, keep_prob=1):
    input_nodes = tf.Dimension(input.get_shape()[3])
    output_shape = tf.stack([input_nodes, -1])
    trans_input = tf.transpose(input, [3,0,1,2]) #[C, N, H, W]
    output = tf.reshape(trans_input, output_shape)#[C, N*H*W]
    if keep_prob != 1:
        output = tf.nn.dropout(output, keep_prob=keep_prob)
    return output

def fully_connected(input_, output_dim, activation_fn = tf.nn.relu, drop_out=True, name="fully_connected"):
    with tf.variable_scope(name):
        #input_ = tf.reshape(input_,np.array([input_.shape[0], -1]))
        fc_output = tf.contrib.layers.fully_connected(input_, output_dim, activation_fn)
        if drop_out:
            fc_output = tf.nn.dropout(fc_output, keep_prob=0.5)
        return fc_output

def ASPP(feature, cls_num, name = "ASPP"):
    with tf.variable_scope(name):
        pre_act_feature = relu(input_ = feature)
        c0 = atrous_conv2d(input_=pre_act_feature, output_dim = cls_num, kernel_size = 3, dilation = 6, padding = "SAME", name = "c0", biased = True)
        c1 = atrous_conv2d(input_=pre_act_feature, output_dim=cls_num, kernel_size=3, dilation=12, padding="SAME", name="c1",
                           biased=True)
        c2 = atrous_conv2d(input_=pre_act_feature, output_dim=cls_num, kernel_size=3, dilation=18, padding="SAME", name="c2",
                           biased=True)
        c3 = atrous_conv2d(input_=pre_act_feature, output_dim=cls_num, kernel_size=3, dilation=24, padding="SAME", name="c3",
                           biased=True)
        output = tf.add_n(inputs=[c0, c1, c2, c3])
        return output

def transfer_network(feature, semantic_vector_length, cls_nums, reuse=False, name="transfer_network"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        pre_norm = batch_norm_slim(feature, is_training=True, activation_fn=tf.nn.relu, name="pre_norm")
        c0 = conv2d(input_=pre_norm, output_dim=semantic_vector_length, kernel_size=1, stride=1, name="c0", biased=True)
        c0_channel = tf.Dimension(c0.get_shape()[3])
        c0_shape = tf.stack([c0_channel, -1])
        reshaped_c0 = tf.reshape(tf.transpose(c0, [3,0,1,2]), c0_shape) # n * (H*W)
        c1 = conv2d(input_=pre_norm, output_dim=cls_nums, kernel_size=1, stride=1, name="c1", biased=True)
        c1_channel = tf.Dimension(c1.get_shape()[3])
        c1_shape = tf.stack([-1, c1_channel])
        reshaped_c1 = tf.reshape(c1, c1_shape)  # (H*W) * cls
        restore_vec = tf.matmul(reshaped_c0, reshaped_c1) # n * cls

        c2 = conv2d(input_=pre_norm, output_dim=semantic_vector_length, kernel_size=1, stride=1, name="c2", biased=True)
        c2_channel = tf.Dimension(c2.get_shape()[3])
        c2_shape = tf.stack([-1, c2_channel])
        reshaped_c2 = tf.reshape(c2, c2_shape)  # (H*W) * n
        addition_factor = tf.nn.softmax(tf.matmul(reshaped_c2, restore_vec), -1) # (H*W) * cls

        c3 = conv2d(input_=pre_norm, output_dim=cls_nums, kernel_size=1, stride=1, name="c3", biased=True)
        c3_shape = tf.shape(c3)
        reshaped_addition_factor = tf.reshape(addition_factor, c3_shape)
        alpha_factor = make_var(name="alpha_factor", shape=[1], trainable=True)
        feature_addition = alpha_factor * reshaped_addition_factor * c3

        ASPP_out = ASPP(feature, cls_nums, name = "ASPP")

        ASPP_with_addition = tf.add(feature_addition, ASPP_out)
        # n*cls, 1*H*W*cls
        return restore_vec, ASPP_with_addition